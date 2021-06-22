import random
import pandas as pd
import numpy as np
from .hash_utils import ModHashGenerator

class StreamWalkUpdater():
    """
    Sample temporal random walks for the StreamWalk algorithm
    
    Parameters
    ----------
    half_life : int
        Half-life in seconds for time decay
    max_len : int
        Maximum length of the sampled temporal random walks
    beta : float
        Damping factor for long paths
    cutoff: int
        Temporal cutoff in seconds to exclude very distant past
    k: int
        Number of sampled walks for each edge update
    full_walks: bool
        Return every node of the sampled walk for representation learning (full_walks=True) or only the endpoints of the walk (full_walks=False)
    """
    def __init__(self, half_life=7200,  max_len=3, beta=0.9, cutoff=604800, k=4, full_walks=False):
        self.c = - np.log(0.5) / half_life
        self.beta = beta
        self.half_life = half_life
        self.k = k
        self.cutoff  = cutoff
        self.max_len = max_len
        self.full_walks = full_walks
        self.G = {}
        self.times = {}
        self.cent = {}
        self.cent_now = {}
        self.lens = {}
        for j in range(max_len):
            self.lens[j+1] = 0
        
    def __str__(self):
        return "streamwalk_hl%i_ml%i_beta%.2f_cutoff%i_k%i_fullw%s" % (self.half_life, self.max_len, self.beta, self.cutoff, self.k, self.full_walks)

    def process_new_edge(self, src, trg, time):
        self.update(src, trg, time)
        return self.sample_node_pairs(src, trg, time, self.k)
        
    def sample_node_pairs (self, src, trg, time, sample_num):
        if src not in self.G:
            # src is not reachable from any node within cutoff
            return [(src, trg)] * sample_num
        edge_tuples = [(src, trg, time)] * sample_num
        pairs = [self.sample_single_walk(tup) for tup in edge_tuples]
        return pairs

    def sample_single_walk(self, edge_tuple):
        src, trg, time = edge_tuple
        node_, time_, cent_ = src, self.times[src], self.cent[src]
        walk = []
        walk.append(node_)
        while True:
            if random.uniform(0, 1) < 1 / (cent_ * self.beta + 1) or (node_ not in self.G) or len(walk) >= self.max_len:
                break
            sum_ = cent_ * random.uniform(0, 1)
            sum__ = 0
            broken = False
            for (n, t, c) in reversed(self.G[node_]):
                if t < time_:
                    sum__ += (c * self.beta + 1) * np.exp(self.c * (t - time_))
                    if sum__ >= sum_:
                        broken = True
                        break
            if not broken:
                break
            node_, time_, cent_ = n, t, c
            walk.append(node_)
        self.lens[len(walk)] += 1
        if(self.full_walks):
            return [trg]+walk
        else:
            return (node_,trg)

    def update(self, src, trg, time):
        # apply time decay for trg
        if trg in self.cent:
            self.cent[trg] = self.cent[trg] * np.exp(self.c * (self.times[trg] - time))
        else:
            self.cent[trg] = 0
        src_cent = 0
        if src in self.times:
            src_cent = self.cent[src]
            if self.times[src] < time:
                src_cent = src_cent * np.exp(self.c * (self.times[src] - time))
                # update centrality and time for src
                self.cent[src] = src_cent
                self.times[src] = time
                self.cent_now[src] = 0
                self.clean_in_edges(src, time)
            else:
                # if src is currently active then adjust centrality
                src_cent = src_cent - self.cent_now[src]
        self.cent[trg] += src_cent * self.beta + 1
        if (trg not in self.times) or (self.times[trg] < time):
            # cent_now is initialized for each node in each second
            self.cent_now[trg] = 0
        self.cent_now[trg] += src_cent * self.beta + 1
        # collect recent edges for each vertex
        if trg not in self.G:
            self.G[trg] = []
        self.G[trg].append((src, time, src_cent))
        self.times[trg] = time
        # clean in egdes
        self.clean_in_edges(trg, time)
    
    def clean_in_edges(self, node, time):
        ii = 0
        for (s, t, c) in self.G[node]:
            if time - t < self.cutoff:
                break
            ii += 1
        # drop old inedges
        self.G[node] = self.G[node][ii:]
    

class SecondOrderUpdater():
    """
    Sample node pairs for the online second order similarity algorithm
    
    Parameters
    ----------
    half_life : int
        Half-life in seconds for time decay
    num_hash : int
        Number of hash functions to use for similarity approximation
    hash_gen : object
        Hash function generator class. Choose from the implemented generators in `online_node2vec.online.hash_utils`
    in_edges: float 
        Weight of in-neighborhood fingerprint
    out_edges: float
        Weight of out-neighborhood fingerprint
    incr_condition: bool
        Enable strict fingerprint matching criteria
    """
    def __init__(self, half_life=7200, num_hash=20, hash_generator=ModHashGenerator(), in_edges=0.0, out_edges=1.0, incr_condition=True):
        # parameters
        self.half_life = half_life
        self.num_hash = num_hash
        self.hash_gen = hash_generator
        self.in_edges = in_edges
        self.out_edges = out_edges
        self.incr_condition = incr_condition
        # variables
        self.c = - np.log(0.5) / half_life
        self.hash_functions = self.hash_gen.generate(self.num_hash)
        self.out_fingerprint_data = {}
        self.out_edgelist_graph = {}
        self.in_fingerprint_data = {}
        self.in_edgelist_graph = {}
        self.extended_chosen_list = []

        self.sampled = None

    def __str__(self):
        return "secondorder_hl%i_numh%i_%s_in%.2f_out%.2f_incr%s" % (self.half_life, self.num_hash, str(self.hash_gen), self.in_edges, self.out_edges, self.incr_condition)

    def process_new_edge(self, src, trg, now):
        sampled = []
        if self.in_edges > 0 or self.out_edges > 0:
            # directed case
            if self.in_edges > 0:
                sampled += self.update_second_order(trg, src, now, self.in_edgelist_graph, self.in_fingerprint_data, self.in_edges)
            if self.out_edges > 0:
                sampled += self.update_second_order(src, trg, now, self.out_edgelist_graph, self.out_fingerprint_data, self.out_edges)
        else:
            # undirected case
            sampled += self.update_second_order(trg, src, now, self.out_edgelist_graph, self.out_fingerprint_data)
            sampled += self.update_second_order(src, trg, now, self.out_edgelist_graph, self.out_fingerprint_data)
        return sampled

    def update_second_order(self, u, v, now, edgelist_graph, fingerprint_data,  proba=0.5):
        "Update fingerprints then sample node pairs for word2vec"

        sampled_node_pairs = []
        # update second order similarities (update every fingerprints for node v")
        self.heuristic_update(v, u, v, now, fingerprint_data)
        neighbors = edgelist_graph.get(u, set([]))
        for (x,t) in list(neighbors):
            neighbors.remove((x,t))
            rnd = np.random.random()
            if(rnd > np.exp(-self.c * (now - t))) or (rnd > proba):
                continue
            else:
                neighbors.add((x,now))
            if x != v:
                # update every fingerprints for node x
                self.heuristic_update(x, u, v, now, fingerprint_data)
                # sample node pairs for word2vec
                for i in range(self.num_hash):
                    fp_v = fingerprint_data[v][i][1]
                    fp_x = fingerprint_data[x][i][1]
                    if self.incr_condition:
                        # fingerprint must be u for both nodes
                        match_condition = fp_v == u and fp_x == u
                    else:
                        # any common fingerprint value match is valid
                        match_condition = fp_v == fp_x and (fp_v is not None)
                    if match_condition:
                        sampled_node_pairs.append((v, x))  # string casting for word2vec
                        self.extended_chosen_list.append({
                            'edge_t': now,
                            'edge_src': u,
                            'edge_trg': v,
                            'sample_x': v,
                            'sample_y': x,
                            'method': 'secOrdSim'
                        })
        # update neighborhoods
        neighbors.add((v,now))
        edgelist_graph[u] = neighbors
        return sampled_node_pairs

    def heuristic_update(self, x, u, v, now, fingerprint_data):
        "Update every fingerprints for node x"

        if x in fingerprint_data:
            all_fingerprint_item = fingerprint_data[x]
            updated_fingerprint_items = [
                self.heuristic_update_base(x, u, v, now, fp_item)
                for fp_item in all_fingerprint_item
            ]
        else:
            new_fp = u if x == v else None
            updated_fingerprint_items = [(i, new_fp, now) for i in range(self.num_hash)]

        fingerprint_data[x] = updated_fingerprint_items

    def heuristic_update_base(self, x, u, v, now, fp_item):
        "Update single fingerprint for node x"
        fp_idx, fp, last_update = fp_item  # extract old fingerprint item

        rand = np.random.random()
        keep_proba = np.exp(-self.c * (now - last_update))
        dice_roll = rand > keep_proba

        new_fp = fp
        if x == v:
            if (fp is None) or dice_roll:
                # unset or old fingerprint
                new_fp = u
            else:
                # minhash rule for recent fingerprint
                hash_fp = self.hash_functions[fp_idx](int(fp))
                hash_u = self.hash_functions[fp_idx](int(u))
                if hash_u < hash_fp:
                    new_fp = u
        elif dice_roll:
            # drop old fingerprint
            new_fp = None

        return (fp_idx, new_fp, now)
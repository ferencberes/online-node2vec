import random
import pandas as pd
import numpy as np
from .hash_utils import ModHashGenerator

def update_walk_instance_weight(decay, instance):
        src, length, weight = instance
        return (src, length, weight*decay)
    
def get_num_samples(x, k=3, K=50, c=0.5):
    """"Transformed sigmoid function"""
    if k <= 0 or K <= 0:
        raise RuntimeError("Parameter 'K' and 'k' must be positive!")
    b = k / (K-k)
    a = b * K
    return int(np.floor(a / (b + np.exp(-c * x))))

class TemporalWalkUpdater():
    """Temporal Walk Algorithm"""
    def __init__(self, half_life=7200, window_size=3, k=4, K=50, gamma=0.5, max_num_walks_per_node=1000, p=0.9):
        #parameters
        self.half_life = half_life
        self.c = - np.log(0.5) / half_life
        self.window_size = window_size
        self.max_num_walks_per_node = max_num_walks_per_node
        self.p = p
        self.k = k
        self.K = K
        self.gamma = gamma
        #variables
        self.trg_index = {}
        self.src_index = {}
        self.trg_last_updated = {}
        self.num_stored_walks = 0
        self.extended_chosen_list = []
        
    def __str__(self):
        return "tempwalk_hl%i_win%i_k%i_K%i_gamma%0.2f_mnw%i_p%0.2f" % (self.half_life, self.window_size, self.k, self.K, self.gamma, self.max_num_walks_per_node, self.p)
                    
    def sample_node_pairs(self, src, trg, time):
        """Sample walk starting nodes based on weights"""
        sampled_pairs = [(src,trg)]
        if self.k > 0:
            sample_df = pd.DataFrame(self.trg_index[trg], columns=["src","length","weight"])
            sum_weights = sample_df["weight"].sum()
            num_samples = get_num_samples(sum_weights, k=self.k, K=self.K, c=self.c)
            chosen_df = sample_df.sample(n=num_samples, replace=True, weights="weight")
            chosen_df["trg"] = trg
            sampled_pairs += list(zip(chosen_df["src"],chosen_df["trg"]))
            # include extended information
            chosen_df["edge_t"] = time
            chosen_df["edge_src"] = src
            chosen_df["edge_trg"] = trg
            chosen_df["method"] = 'tempWalk'
            chosen_df = chosen_df.rename(index=str, columns={"src": "sample_x", "trg": "sample_y"})
            self.extended_chosen_list += list(chosen_df.to_dict(orient="index").values())
        return sampled_pairs
    
    def process_new_edge(self, src, trg, time):
        """Updates the stored walks based on the current edge (src,trg,time)"""
        # load stored walks
        src_walks = self.trg_index.get(src,[])
        trg_walks = self.trg_index.get(trg,[])
        self.num_stored_walks -= len(trg_walks)
        # update weights of stored walks
        if len(src_walks) > 0:
            src_walks = self.update_walk_weights(src_walks, src, time)
            self.trg_index[src] = src_walks
        if len(trg_walks) > 0:
            trg_walks = self.update_walk_weights(trg_walks, trg, time)
        # append new walk with length beta
        trg_walks.append((src,1,1.0))
        # pandas implementation seems to be faster
        updated_trg_walks = self.update_walks_instances_with_df(src_walks, trg_walks, trg)
        self.num_stored_walks += len(updated_trg_walks)
        self.trg_index[trg] = updated_trg_walks
        return self.sample_node_pairs(src, trg, time)
        
    def update_walks_instances_with_df(self, src_walks, trg_walks, target):
        """walk updates using only pandas on a single thread"""
        src_df = pd.DataFrame(src_walks, columns=["start","length","weight"])
        trg_df = pd.DataFrame(trg_walks, columns=["start","length","weight"])
        if len(src_df) > 0:
            # drop loops & long walks
            src_df = src_df[(src_df["start"] != target) & (src_df["length"] < self.window_size)]
            src_df["length"] = src_df["length"] + 1
            all_walks = pd.concat([src_df,trg_df])
            summed_walks = all_walks.groupby(by=["start","length"])["weight"].sum().reset_index()
        else:
            summed_walks = trg_df
        # sample walks if size exceeds limit
        if len(summed_walks) > self.max_num_walks_per_node:
            summed_walks = summed_walks.sample(n=int(self.p*self.max_num_walks_per_node), weights="weight")
        walk_instances = list(zip(summed_walks["start"],summed_walks["length"],summed_walks["weight"]))
        return walk_instances
        
    def update_walk_weights(self, walks, trg_node, now):
        """Execute lazy time decay update for the stored walk weights"""
        last_update = self.trg_last_updated.get(trg_node, now)
        decay = np.exp(-self.c * (now-last_update))
        updated_walks = [update_walk_instance_weight(decay, w) for w in walks]
        self.trg_last_updated[trg_node] = now
        return updated_walks

    
class TemporalRandomWalkUpdater():
    """Temporal (Random) Walk Algorithm"""
    def __init__(self, half_life=7200,  max_len = 3, sample_num=4, beta=0.2, cutoff = 604800, full_walks = False):
        self.c = - np.log(0.5) / half_life
        self.beta = beta
        self.half_life = half_life
        self.sample_num = sample_num
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
        self.extended_chosen_list = []
        
    def __str__(self):
        return "temprandwalk_hl%i_ml%i_beta%.2f_sn%i_cutoff%i_fullw%s" % (self.half_life, self.max_len, self.beta, self.sample_num, self.cutoff, self.full_walks)

    def sample_node_pairs (self, src, trg, time):
        pairs = []
        if src not in self.G:
            # src is not reachable from any node within cutoff
            return [(src,trg)]*self.sample_num
        for i in range(self.sample_num):
            node_ , time_, cent_ = src, self.times[src], self.cent[src]
            walk = []
            while True:
                walk.append(node_)
                if random.uniform(0,1) < 1/(cent_ * self.beta + 1) or (node_ not in self.G) or len(walk)>=self.max_len:  break
                sum_ = cent_ * random.uniform(0,1); sum__ = 0
                for (n,t,c) in reversed(self.G[node_]):
                    if t < time_:
                        sum__ += (c * self.beta + 1)* np.exp( self.c * (t - time_) )
                        if sum__ >= sum_: break
                node_,time_,cent_ = n,t,c
            self.lens[len(walk)]+=1
            if(self.full_walks):
                pairs.append([trg]+walk)
            else:
                pairs.append((node_,trg))
                self.extended_chosen_list.append({
                'sample_x' : node_,
                'sample_y' : trg,
                'method' : 'tempRandWalk',
                'edge_t' : time,
                'edge_src' : src,
                'edge_trg' : trg,  
            })
        return pairs

    def process_new_edge(self, src, trg, time):
        # apply time decay for trg
        if trg in self.cent:
            self.cent[trg] = self.cent[trg]*np.exp(self.c*(self.times[trg]-time))
        else:
            self.cent[trg] = 0
        src_cent =  0
        if src in self.times:
            src_cent = self.cent[src]
            if self.times[src] < time:
                # apply time decay for src
                src_cent = src_cent * np.exp(self.c*(self.times[src]-time))
            else:
                # if src is currently active then adjust centrality
                src_cent = src_cent - self.cent_now[src]
        self.cent[trg] += src_cent * self.beta + 1
        if (trg not in self.times) or (self.times[trg] < time):
            # cent_now is initialized for each node in each second
            self.cent_now[trg] = 0
        self.cent_now[trg] += src_cent * self.beta + 1
        if trg not in self.G:
            self.G[trg] = []
        # collect recent edges for each vertex
        self.G[trg].append((src,time,src_cent))
        ii = 0
        for (s,t,c) in self.G[trg]:
            if time - t < self.cutoff: break
            ii+=1
        # drop old inedges
        self.G[trg] = self.G[trg][ii:]
        # update node activation time
        self.times[trg] = time
        # generate node pairs for training
        return self.sample_node_pairs(src, trg, time)

    
class OnlineSecondOrderSim():
    """Temporal Neighborhood Algorithm"""
    def __init__(self, half_life=7200, num_hash=20, hash_generator=ModHashGenerator(), real_direction=False):
        # parameters
        self.half_life = half_life
        self.num_hash = num_hash
        self.hash_gen = hash_generator
        self.real_direction = real_direction
        # variables
        self.c = - np.log(0.5) / half_life
        self.hash_functions = self.hash_gen.generate(self.num_hash)
        self.fingerprint_data = {}
        self.out_edgelist_graph = {}
        # self.in_edgelist_graph = {}
        self.extended_chosen_list = []

    def __str__(self):
        return "secondorder_hl%i_numh%i_%s_rdir%s" % (self.half_life, self.num_hash, str(self.hash_gen), self.real_direction)

    def process_new_edge(self, src, trg, now):
        return self.update_second_order(src, trg, now, self.out_edgelist_graph)

    def update_second_order(self, u, v, now, out_edgelist_graph):
        "Update fingerprints then sample node pairs for word2vec"

        sampled_node_pairs = []
        # update second order similarities
        self.heuristic_update(v, u, v, now)
        neighbors = out_edgelist_graph.get(u, set([]))
        for x in neighbors:
            if x != v:
                self.heuristic_update(x, u, v, now)
                # sample node pairs for word2vec
                for i in range(self.num_hash):
                    fp_v = self.fingerprint_data[v][i][1]
                    fp_x = self.fingerprint_data[x][i][1]
                    if fp_v == fp_x and (fp_v is not None):
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
        neighbors.add(v)
        out_edgelist_graph[u] = neighbors
        return sampled_node_pairs

    def heuristic_update(self, x, u, v, now):
        "Update every fingerprints for node x"

        if x in self.fingerprint_data:
            all_fingerprint_item = self.fingerprint_data[x]
            updated_fingerprint_items = [
                self.heuristic_update_base(x, u, v, now, fp_item)
                for fp_item in all_fingerprint_item
            ]
        else:
            new_fp = u if x == v else None
            updated_fingerprint_items = [(i, new_fp, now) for i in range(self.num_hash)]

        self.fingerprint_data[x] = updated_fingerprint_items

    def heuristic_update_base(self, x, u, v, now, fp_item):
        "Update single fingerprint for node x"
        fp_idx, fp, last_update = fp_item  # extract old fingerprint item

        rand = np.random.random()
        keep_proba = np.exp(-self.c * (now - last_update))
        dice_roll = rand > keep_proba

        new_fp = fp
        if x == v:
            if (fp is None) or dice_roll:
                new_fp = u
            else:
                hash_fp = self.hash_functions[fp_idx](int(fp))
                hash_u = self.hash_functions[fp_idx](int(u))
                if hash_u < hash_fp:
                    new_fp = u
        elif dice_roll:
            new_fp = None

        return (fp_idx, new_fp, now)
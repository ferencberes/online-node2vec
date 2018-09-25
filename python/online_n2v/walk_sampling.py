import multiprocessing, functools
import pandas as pd
import numpy as np

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
    def __init__(self, half_life=7200, window_size=3, k=4, K=50, gamma=0.5, max_num_walks_per_node=1000, p=0.9, n_threads=1):
        #parameters
        self.half_life = half_life
        self.c = - np.log(0.5) / half_life
        self.window_size = window_size
        self.max_num_walks_per_node = max_num_walks_per_node
        self.p = p
        self.k = k
        self.K = K
        self.gamma = gamma
        self.n_threads = n_threads
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
            sample_df["weight"] /= self.sum_weights
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
        if self.n_threads > 1 and len(walks) > self.n_threads:
            f_partial = functools.partial(update_walk_instance_weight, decay)
            pool = multiprocessing.Pool(processes=self.n_threads)
            updated_walks = pool.map(f_partial, walks)
            pool.close()
            pool.join()
        else:
            updated_walks = [update_walk_instance_weight(decay, w) for w in walks]
        self.trg_last_updated[trg_node] = now
        return updated_walks

import sqlite3
class TemporalWalkUpdaterSQLite():
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
        self.num_stored_walks = 0
        self.extended_chosen_list = []
        
        self.sqlite = sqlite3.connect(':memory:')
        self.sqlite.execute("DROP TABLE IF EXISTS walks;")
        self.sqlite.execute("""CREATE TABLE walks (
            src VARCHAR(32) NOT NULL,
            trg VARCHAR(32) NOT NULL,
            length INTEGER NOT NULL,
            weight DOUBLE NOT NULL,
            time INTEGER NOT NuLL
        )""")
        self.sqlite.execute("CREATE INDEX walk_trgs_index ON walks (trg);")
        # update weight: multiplying by decay
        self.sqlite.create_function("node_weight", 3, lambda now, last_update, weights: weights * np.exp(- self.c * (now - last_update)))
        self.sqlite.create_function("update_time", 1, lambda now: now)
        self.sqlite.create_function("sampled_node_weight", 2, lambda now, etime: np.random.rand() ** (1 / np.exp(- c * (now - etime))))
        # Domi magic?
        self.sqlite.create_function("sampled_weight_score", 1, lambda w: np.random.rand() ** (1 / w))
        
    def __str__(self):
        return "sqltempwalk_hl%i_win%i_k%i_K%i_gamma%0.2f_mnw%i_p%0.2f" % (self.half_life, self.window_size, self.k, self.K, self.gamma, self.max_num_walks_per_node, self.p)
    
    def sample_node_pairs(self, src, trg, time):
        """Sample walk starting nodes based on weights"""
        srcs = [src]
        if self.k > 0:
            walks = list(zip(*list(self.sqlite.execute("""
                SELECT src, weight FROM walks
                WHERE trg = "%s"
            """ % trg))))
            weights_nonorm = np.array(walks[1])
            weights_sum = np.sum(weights_nonorm)
            weights = weights_nonorm / weights_sum
            num_samples = get_num_samples(weights_sum, k=self.k, K=self.K, c=self.c)
            srcs = list(np.random.choice(walks[0], num_samples, replace=True, p=weights))
            # include extended information
            chosen_pairs = []
            for x in srcs:
                chosen_pairs.append({
                    'sample_x' : x,
                    'sample_y' : trg,
                    'method' : 'tempWalk',
                    'edge_t' : time,
                    'edge_src' : src,
                    'edge_trg' : trg,  
                })
            self.extended_chosen_list += chosen_pairs
        return list(zip(srcs, np.repeat(trg, len(srcs))))
    
    def process_new_edge(self, src, trg, time):
        # walks ending in target to temporary table
        self.sqlite.execute("DROP TABLE IF EXISTS trg_walks;")
        self.sqlite.execute("""
            CREATE TEMPORARY TABLE trg_walks AS
            SELECT * FROM walks WHERE walks.trg = "%s";
        """ % trg)
        
        # insert current edge
        self.sqlite.execute(
            "INSERT INTO trg_walks (src, trg, length, weight, time) values (?,?,?,?,?)",
            (src, trg, 1, 1.0, int(time))
        )
    
        # insert walks ending in source
        self.sqlite.execute("""
            INSERT INTO trg_walks
            SELECT src, "%s", length+1 as nlen, weight, time FROM walks WHERE
                walks.trg = "%s" and
                src != "%s" and
                nlen <= %i;
        """ % (trg, src, trg, self.window_size))
        
        self.sqlite.execute("UPDATE trg_walks SET weight = node_weight(%i, trg_walks.time, trg_walks.weight)" % time)
        self.sqlite.execute("UPDATE trg_walks SET time = update_time(%i)" % time)
        
        # groupby
        self.sqlite.execute("DROP TABLE IF EXISTS trg_walks_2;")
        self.sqlite.execute("""
            CREATE TEMPORARY TABLE trg_walks_2 AS
            SELECT src, trg, length, SUM(weight) as weight, time
            FROM trg_walks
            GROUP BY src;
        """)
        
        # delete original walks
        self.sqlite.execute("""DELETE FROM walks WHERE trg = "%s" """ % trg)

        # load new walks back into table
        target_walks_num = self.sqlite.execute("SELECT count(*) from trg_walks_2;").fetchone()[0]
        #print(5, target_walks_num)
        if target_walks_num > self.max_num_walks_per_node:
            # sample without replacement according to weights
            # https://stackoverflow.com/a/18282419/336403
            self.sqlite.execute("""
                INSERT INTO walks
                SELECT * FROM trg_walks_2
                ORDER BY sampled_weight_score(trg_walks_2.weight) DESC
                LIMIT %f
            """ % self.p*self.max_num_walks_per_node)
        else:
            self.sqlite.execute("""
                INSERT INTO walks
                SELECT * FROM trg_walks_2
            """)
        self.num_stored_walks = self.sqlite.execute("SELECT count(*) from walks;").fetchone()[0]
        return self.sample_node_pairs(src, trg, time)
    
class OnlineSecondOrderSim():
    def __init__(self, hash_functions, half_life=7200, real_direction=False, n_threads=1):
        # parameters
        self.hash_functions = hash_functions
        self.half_life = half_life
        self.real_direction = real_direction
        self.c = - np.log(0.5) / half_life
        self.n_threads = n_threads
        # variables
        self.k = len(hash_functions)
        self.fingerprint_data = {}
        self.out_edgelist_graph = {}
        # self.in_edgelist_graph = {}
        self.extended_chosen_list = []
        self.num_stored_walks = None # this parameter is needed due to information logging

    def __str__(self):
        return "secondorder_hl%i_numh%i_rdir%s" % (self.half_life, self.k, self.real_direction)

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
                for i in range(self.k):
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
            updated_fingerprint_items = [(i, new_fp, now) for i in range(self.k)]

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

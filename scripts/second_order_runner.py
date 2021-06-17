# coding: utf-8
import concurrent.futures
import os, sys, time
import pandas as pd
from online_node2vec.online.w2v_learners import GensimWord2Vec, OnlineWord2Vec
from online_node2vec.online.walk_sampling import SecondOrderUpdater
from online_node2vec.online.online_node2vec_models import LazyNode2Vec, OnlineNode2Vec
from online_node2vec.online import hash_utils as hu
from online_node2vec.data.tennis_handler import load_edge_data

output_folder = "../results/"
data_dir = "../data/"
data_id = "rg17"
delta_time = 3600*6

# updater parameters
is_online = True
is_decayed = True
incr_condition = True

half_life = 43200
hash_num = 20
hash_type = "mod"
in_edges = 0.0
out_edges = 1.0

# learner parameters
dim = 128
lr_rate = 0.01
neg_rate = 5
uniform_ratio = 0.8
interval = 86400
temp_noise = False
loss = "square"

def generate_embeddings(sample_id):

    if hash_type == "mod":
        hash_gen = hu.ModHashGenerator()
    elif hash_type == "mul":
        hash_gen = hu.MulHashGenerator()
    elif hash_type == "map":
        hash_gen = hu.MapHashGenerator()
    else:
        raise RuntimeError("Invalid hash config!")

    updater = SecondOrderUpdater(half_life=half_life, num_hash=hash_num, hash_generator=hash_gen, in_edges=in_edges, out_edges=out_edges, incr_condition=incr_condition)

    if not is_online:
        learner = GensimWord2Vec(embedding_dims=dim, lr_rate=lr_rate, sg=1, neg_rate=neg_rate, n_threads=4)
        # initialize node2vec object
        online_n2v = LazyNode2Vec(updater, learner, is_decayed)
    else:
        learner = OnlineWord2Vec(embedding_dims=dim, loss=loss, lr_rate=lr_rate, neg_rate=neg_rate, window=0, interval=interval, temporal_noise=temp_noise, use_pairs=True, uniform_ratio=uniform_ratio)
        # initialize node2vec object
        online_n2v = LazyNode2Vec(updater, learner, is_decayed)
        #online_n2v = OnlineNode2Vec(updater, learner, is_decayed)

    root_dir = "%s/%s/features_%s/delta_%i" % (output_folder, data_id, sample_id, delta_time)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    online_n2v.run(edge_data, delta_time, root_dir, start_time=start_time, end_time=end_time)
    return root_dir

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        num_samples = int(sys.argv[1])
        num_threads = int(sys.argv[2])
        if len(sys.argv) >= 4:
            max_days = int(sys.argv[3])
        else:
            max_days = None
        print(num_samples, num_threads, max_days)
        samples = range(num_samples)
        START = time.time()
        edge_data, start_time, end_time = load_edge_data(data_dir, data_id, max_days)
        if len(samples) > 1:
            executor = concurrent.futures.ProcessPoolExecutor(num_threads)
            root_dirs = list(executor.map(generate_embeddings, samples))
        else:
            root_dirs = list(map(generate_embeddings, samples))
        print("compute done")
    else:
        print("Usage: <num_samples> <max_threads> <max_days?>")
# coding: utf-8
import os, sys, time
import concurrent.futures
import pandas as pd
from online_node2vec.online.w2v_learners import GensimWord2Vec, OnlineWord2Vec
from online_node2vec.online.walk_sampling import StreamWalkUpdater
from online_node2vec.online.online_node2vec_models import LazyNode2Vec, OnlineNode2Vec
from online_node2vec.online import hash_utils as hu
from online_node2vec.data.tennis_handler import load_edge_data

data_dir = "../data/"
output_folder = "../results/"
delta_time = 3600*6

# updater parameters
data_id = "rg17"
half_life = 7200
is_decayed = True
is_online = True

is_fw = False
max_length = 2
k = 4
beta = 0.9
cutoff = 604800

# learner parameters
dim = 128
lr_rate = 0.035
neg_rate = 10
mirror = False
onlymirror = False
init = "uniform"
exportW1 = False
interval = 86400
temp_noise = False
loss = "square"

def generate_embeddings(sample_id):
    updater = StreamWalkUpdater(half_life=half_life, max_len=max_length, beta=beta, cutoff=cutoff, k=k, full_walks=is_fw)
    if not is_online:
        learner = GensimWord2Vec(embedding_dims=dim, lr_rate=lr_rate, sg=1, neg_rate=neg_rate, n_threads=4)
        # initialize node2vec object
        online_n2v = LazyNode2Vec(updater, learner, is_decayed)
    else:
        learner = OnlineWord2Vec(embedding_dims=dim, loss=loss, lr_rate=lr_rate, neg_rate=neg_rate, mirror=mirror, onlymirror=onlymirror, init=init, exportW1=exportW1, window=2, interval=interval, temporal_noise=temp_noise, use_pairs=(not is_fw))
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
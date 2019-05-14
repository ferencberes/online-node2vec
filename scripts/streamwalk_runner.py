# coding: utf-8

import os, sys, time
import pandas as pd
import time

sys.path.insert(0,"../python")

from online_n2v.w2v_learners import GensimWord2Vec, OnlineWord2Vec
from online_n2v.walk_sampling import StreamWalkUpdater
from online_n2v.online_node2vec_models import LazyNode2Vec, OnlineNode2Vec
from online_n2v import hash_utils as hu

import evaluation.distance_computer as distc
import evaluation.ndcg_computer as ndcgc
import data.tennis_handler as th
import data.n2v_embedding_handler as n2veh

output_folder = "../results/"
delta_time = 3600*6

# updater parameters
data_id = "rg17"
half_life = 7200
is_decayed = True
is_online = True

is_fw = False
max_length = 3
k = 4
K = 4
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
    updater = StreamWalkUpdater(half_life=half_life, max_len=max_length, beta=beta, cutoff=cutoff, k=k, K=K, full_walks = is_fw)
    if not is_online:
        learner = GensimWord2Vec(embedding_dims=dim, lr_rate=lr_rate, sg=1, neg_rate=neg_rate, n_threads=4)
        # initialize node2vec object
        online_n2v = LazyNode2Vec(updater, learner, is_decayed)
    else:
        learner = OnlineWord2Vec(embedding_dims=dim, loss=loss, lr_rate=lr_rate, neg_rate=neg_rate, mirror=mirror, onlymirror=onlymirror, init=init, exportW1=exportW1, window=2, interval=interval, temporal_noise=temp_noise, use_pairs=(not is_fw))
        # initialize node2vec object
        online_n2v = LazyNode2Vec(updater, learner, is_decayed)
        #online_n2v = OnlineNode2Vec(updater, learner, is_decayed)

    root_dir = "/%s/%s/features_%s/delta_%i" % (output_folder, data_id, sample_id, delta_time)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    online_n2v.run(edge_data, delta_time, root_dir, start_time=start_time, end_time=end_time)
    return root_dir

if __name__ == "__main__":
    num_samples = 10
    samples = range(num_samples)
    START = time.time()
    
    # data
    if data_id == "rg17":
        edge_data = pd.read_csv("/mnt/idms/temporalNodeRanking/data/filtered_timeline_data/tsv/rg17/rg17_mentions.csv", sep=" ", names=["time","src","trg"])
        start_time = 1495922400 # 2017-05-28 0:00 Paris # rg17
        total_days = 15
    elif data_id == "uo17":
        edge_data = pd.read_csv("/mnt/idms/temporalNodeRanking/data/filtered_timeline_data/tsv/usopen/usopen_mentions.csv", sep=" ", names=["time","src","trg"])
        start_time = 1503892800 # 2017-08-28 0:00 NY # uo17
        total_days = 14
    else:
        raise RuntimeError("Invalid dataset!")
    #total_days = 1
    #total_days = 3
    end_time = start_time + total_days*86400
    
    import concurrent.futures
    
    if len(samples) > 1:
        executor = concurrent.futures.ProcessPoolExecutor(len(samples))
        root_dirs = list(executor.map(generate_embeddings, samples))
    else:
        root_dirs = list(map(generate_embeddings, samples))
    print("compute done")
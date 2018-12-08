import os, sys
import pandas as pd
sys.path.insert(0,"../python")
from online_n2v.w2v_learners import GensimWord2Vec, OnlineWord2Vec
from online_n2v.walk_sampling import *
from online_n2v.online_node2vec_models import LazyNode2Vec, OnlineNode2Vec

data_id = "rg17"
delta_time = 21600
is_online = False

# updater parameters
updater = TemporalWalkUpdater(half_life=21600)
#updater = TemporalRandomWalkUpdater(half_life=21600)

# learner parameters
dim = 128
lr_rate = 0.05
neg_rate = 5

if not is_online:
    learner = GensimWord2Vec(embedding_dims=dim, lr_rate=lr_rate, sg=1, neg_rate=neg_rate, n_threads=1)
    # initialize node2vec object
    online_n2v = LazyNode2Vec(updater, learner)
else:
    learner = OnlineWord2Vec(embedding_dims=dim, loss=loss, lr_rate=lr_rate, neg_rate=neg_rate, window=window, interval=86400)
    # initialize node2vec object
    online_n2v = OnlineNode2Vec(updater, learner)

# data
if data_id == "rg17":
    edge_data = pd.read_csv("../data/rg17_data/raw/rg17_mentions.csv", sep=" ", names=["time","src","trg"])
    start_time = 1495922400 # 2017-05-28 0:00 Paris # rg17
    end_time = start_time + 15*86400 # rg17
elif data_id == "uo17":
    edge_data = pd.read_csv("../data/uo17_data/raw/uo17_mentions.csv", sep=" ", names=["time","src","trg"])
    start_time = 1503892800 # 2017-08-28 0:00 NY # uo17
    end_time = start_time + 14*86400 # uo17
else:
    raise RuntimeError("Invalid dataset!")

root_dir = "../results/%s_delta_%i" % (data_id, delta_time)
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
    
# run experiment
online_n2v.run(edge_data, delta_time, root_dir, start_time=start_time, end_time=end_time)
import os, sys
import pandas as pd
sys.path.insert(0,"../python")
from offline_n2v.offline_node2vec_model import BatchNode2Vec

from datawand.parametrization import ParamHelper
ph = ParamHelper("../", "OfflineN2V", sys.argv)

delta_time = ph.get("delta_time")
lookback_time = ph.get("lookback_time")
n_threads = ph.get("n_threads")

# updater parameters
dim = ph.get("dimension")
window = ph.get("window")
num_walks = ph.get("num_walks")
walk_length = ph.get("walk_length")
p = ph.get("p")
q = ph.get("q")

offline_n2v = BatchNode2Vec(dim, walk_length, num_walks, window, p, q, lookback_time, n_threads=n_threads)

# data
data_id = ph.get("data_id")
if data_id == "rg17":
    edge_data = pd.read_csv("/mnt/idms/temporalNodeRanking/data/filtered_timeline_data/tsv/rg17/rg17_mentions.csv", sep=" ", names=["time","src","trg"])
    start_time = 1495922400 # 2017-05-28 0:00 Paris # rg17
    end_time = start_time + 15*86400 # rg17
elif data_id == "uo17":
    edge_data = pd.read_csv("/mnt/idms/temporalNodeRanking/data/filtered_timeline_data/tsv/usopen/usopen_mentions.csv", sep=" ", names=["time","src","trg"])
    start_time = 1503892800 # 2017-08-28 0:00 NY # uo17
    end_time = start_time + 14*86400 # uo17
else:
    raise RuntimeError("Invalid dataset!")

root_dir = "/mnt/idms/fberes/data/temporalN2V/%s/offline_n2v/features/delta_%i" % (data_id, delta_time)
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
    
# run experiment
offline_n2v.run(edge_data, delta_time, root_dir, start_time, end_time)

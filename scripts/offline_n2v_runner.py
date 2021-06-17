import os, sys, time
import pandas as pd
sys.path.insert(0,"../python")
from online_node2vec.offline.offline_node2vec_model import BatchNode2Vec
from online_node2vec.data.tennis_handler import load_edge_data

output_folder = "../results/"
data_dir = "../data/"
data_id = "rg17"
delta_time = 3600*6

lookback_time = 172800
dim = 128
window = 3
num_walks = 20
walk_length = 3
p = 1.0
q = 1.0
n_threads = 1

def generate_embeddings(sample_id):
    offline_n2v = BatchNode2Vec(dim, walk_length, num_walks, window, p, q, lookback_time, n_threads=n_threads)
    root_dir = "%s/%s/features_%s/delta_%i" % (output_folder, data_id, sample_id, delta_time)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    offline_n2v.run(edge_data, delta_time, root_dir, start_time=start_time, end_time=end_time)
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
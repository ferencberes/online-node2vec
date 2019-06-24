# coding: utf-8
import os, sys, time, concurrent.futures
import pandas as pd
import numpy as np
import time

sys.path.insert(0,"../python")

import evaluation.distance_computer as distc
import evaluation.ndcg_computer as ndcgc
import data.tennis_handler as th
import data.n2v_embedding_handler as n2veh

output_folder = "../results/"
delta_time = 3600*6

# updater parameters
data_id = "rg17"

def evaluate_embeddings(param_item):
    sample_id, root_dir_prefix = param_item
    ndcg_eval_dir = output_folder + "%s/eval_%i/delta_%i/" % (data_id, sample_id, delta_time)
    ndcg_eval_file = output_folder + "%s/%s.csv" % (ndcg_eval_dir, parameters)
    features_dir = output_folder + "%s/features_%i/delta_%i/%s" % (data_id, sample_id, delta_time, parameters)

    # load n2v embeddings
    print("\nLoading embeddings...")
    
    data = n2veh.load_n2v_features(features_dir, delta_time, total_days, player_labels, eval_window, sep=",")
    
    print(len(data[0]), len(data))
    res_dot = ndcgc.parallel_eval_ndcg(data, gen_id_to_account, "-dot", n_threads=4)
    
    all_df = pd.concat(res_dot)
    if not os.path.exists(ndcg_eval_dir):
        os.makedirs(ndcg_eval_dir)
    all_df.to_csv(ndcg_eval_file)
    mean_ndcg = all_df["ndcg"].mean()
    print(sample_id, mean_ndcg)
    return mean_ndcg

if __name__ == "__main__":
    parameters = sys.argv[1]
    num_samples = 1
    num_threads = 4
    samples = range(num_samples)
    START = time.time()
    
    # data
    if data_id == "rg17":
        total_days = 15
    elif data_id == "uo17":
        total_days = 14
    else:
        raise RuntimeError("Invalid dataset!")
    root_dirs = ["%s/%s/features_%s/delta_%i" % (output_folder, data_id, sample_id, delta_time) for sample_id in range(num_samples)]
    eval_window = delta_time

    # load data
    gen_id_to_account, player_labels = th.get_data_info("../data/%s_preprocessed" % data_id)

    param_items = list(zip(samples, root_dirs))
    executor = concurrent.futures.ProcessPoolExecutor(num_threads)
    metrics = list(executor.map(evaluate_embeddings, param_items))

    print()
    print(parameters)
    print("### ELAPSED TIME ###")
    print("%.2f minutes" % ((time.time()-START) / 60))
    print("### METRICS ###")
    print(metrics)
    print("### PERFORMANCE STATS ###")
    print(list(zip(['mean','std','min','max'],[np.mean(metrics), np.std(metrics), np.min(metrics), np.max(metrics)])))
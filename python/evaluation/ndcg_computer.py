import multiprocessing, functools
import pandas as pd
import numpy as np
from .distance_computer import get_topk_similar, get_combined_topk_similar

def dcg(pred_order, relevance_map, k=None):
    if k == None:
        k = len(pred_order)
    else:
        k = min(k, len(pred_order))
    dcg_score = 0.0
    for i in range(k):
        pred_id = pred_order[i]
        dcg_score += float(relevance_map.get(pred_id,0.0)) / np.log(i+2)
    return dcg_score

def ndcg(pred_order, relevance_dict, k=None):
    """'pred_order' contains ids that are already sorted"""
    relevance_df = pd.DataFrame(list(relevance_dict.items()), columns=["id","relevance"])
    relevance_df = relevance_df.sort_values("relevance", ascending=False)
    IDCG = dcg(list(relevance_df["id"]), relevance_dict, k=k)
    DCG = dcg(pred_order, relevance_dict, k=k)
    return DCG / IDCG

def append_snapshot_indices(snapshot_ndcg_list):
    """reindex snapshots for visualization"""
    for idx, df in enumerate(snapshot_ndcg_list):
        df["snapshot_id"] = idx

### single model toplist ###
        
def parallel_eval_ndcg(features, gen_id_to_account, distance_str="euclidean", ndcg_k=100, n_threads=1):
    if n_threads > 1:
        f_partial = functools.partial(eval_ndcg_for_daily_players, gen_id_to_account, distance_str, ndcg_k)
        pool = multiprocessing.Pool(processes=n_threads)
        res = pool.map(f_partial, features)
        pool.close()
        pool.join()
    else:
        res = [eval_ndcg_for_daily_players(gen_id_to_account, distance_str, ndcg_k, df) for df in features]
    append_snapshot_indices(res)
    return res

def eval_ndcg_for_daily_players(gen_id_to_account, distance_str, ndcg_k, df):
    snapshot_idx, feats, daily_players = df
    ndcg_values = []
    #print("len relevance", len(daily_players))
    for ref_id in daily_players:
        if ref_id in list(feats[0]):
            sims = get_topk_similar(ref_id, df, distance_str, gen_id_to_account, k=None, verbose=False)
            prediction_order = list(sims["id"])
            ndcg_score = ndcg(prediction_order, daily_players, k=ndcg_k)
            ndcg_values.append([ref_id, ndcg_score])
        else:
            # unseen player are assigned 0.0 NDCG
            ndcg_values.append([ref_id, 0.0])
    ndcg_df = pd.DataFrame(ndcg_values, columns=["id","ndcg"])
    ndcg_df["account"] = ndcg_df["id"].apply(lambda x: gen_id_to_account[x])
    ndcg_df["distance"] = distance_str
    ndcg_df["ndcg_k"] = ndcg_k
    return ndcg_df[["id","account","ndcg","distance","ndcg_k"]].sort_values("ndcg", ascending=False)

### combined model toplist ###

def parallel_combined_eval_ndcg(features1, features2, alpha, gen_id_to_account, distance_str="euclidean", ndcg_k=100, n_threads=1):
    feature_pairs = list(zip(features1, features2))
    if n_threads > 1:
        f_partial = functools.partial(eval_ndcg_for_combined_daily_players, gen_id_to_account, distance_str, ndcg_k, alpha)
        pool = multiprocessing.Pool(processes=n_threads)
        res = pool.map(f_partial, feature_pairs)
        pool.close()
        pool.join()
    else:
        res = [eval_ndcg_for_combined_daily_players(gen_id_to_account, distance_str, ndcg_k, alpha, pair) for pair in feature_pairs]
    append_snapshot_indices(res)
    return res

def eval_ndcg_for_combined_daily_players(gen_id_to_account, distance_str, ndcg_k, alpha, dfs):
    df1, df2 = dfs
    snapshot_idx, feats1, daily_players = df1
    _, feats2, _ = df2
    ndcg_values = []
    #print("len relevance", len(daily_players))
    for ref_id in daily_players:
        sims_df = get_combined_topk_similar(ref_id, alpha, df1, df2, distance_str, gen_id_to_account, ndcg_k, False)
        if len(sims_df) > 0:
            prediction_order = list(sims_df["id"])
            ndcg_score = ndcg(prediction_order, daily_players, k=ndcg_k)
            ndcg_values.append([ref_id, ndcg_score])
        else:
            ndcg_values.append([ref_id, 0.0])
    ndcg_df = pd.DataFrame(ndcg_values, columns=["id","ndcg"])
    ndcg_df["account"] = ndcg_df["id"].apply(lambda x: gen_id_to_account[x])
    ndcg_df["distance"] = distance_str
    ndcg_df["ndcg_k"] = ndcg_k
    return ndcg_df[["id","account","ndcg","distance","ndcg_k"]].sort_values("ndcg", ascending=False)
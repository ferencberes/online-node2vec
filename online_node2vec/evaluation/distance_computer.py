from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import numpy as np
import pandas as pd

def metric_for_similarity_f(b, metric="euclidean"):
    if metric == "euclidean":
        return lambda a:np.sqrt(np.sum((a-b)**2))#L2
    elif metric == "cosine":
        return lambda a:cosine(a,b)#cosine distance
    elif metric == "-dot":
        return lambda a:- np.dot(a,b)
    elif metric == "1-pearson":
        return lambda a:1.0 - pearsonr(a,b)[0]
    else:
        raise RuntimeError("Invalid metric!")

#def display(simple_array, num):
#    k = len(simple_array);
#    plt.figure(num);
#    plt.plot(range(k), simple_array);

def get_topk_similar(reference_id, data_part, metric, gen_id_to_account, k=10, verbose=True):
    snapshot_id, feats, relevance_labels = data_part
    snapshot_df = feats.set_index(0)
    row_count = snapshot_df.shape[0]
    distances = []
    #compute each distance

    rep_of_reference =  np.array(snapshot_df.loc[reference_id])
    indices = snapshot_df.index
    metric_f = metric_for_similarity_f(rep_of_reference, metric)
    for idx, row in enumerate(snapshot_df.values):
        if indices[idx] != reference_id:
            distance = metric_f(row)
            distances.append( (indices[idx], distance) )
    rdf = pd.DataFrame(distances, columns=["id","dist"])
    rdf["label"] = rdf["id"].apply(lambda x: relevance_labels.get(x, 0.0))
    rdf = rdf.sort_values('dist')
    if k != None:
        rdf = rdf.iloc[:k]
    if verbose:
        topk_item = list(zip(rdf["id"],rdf["label"],rdf["dist"]))
        print(gen_id_to_account[reference_id])
        i = 1
        for id_, label_, dist_ in topk_item:
            print(i, '\t account name: ', gen_id_to_account[id_], '\t label: ', label_, '\t distance: ', dist_)
            i += 1
    return rdf

def get_combined_topk_similar(ref_id, alpha, df1, df2, distance_str, gen_id_to_account, k, normalize, verbose=False):
    snapshot_idx, feats1, daily_players = df1
    _, feats2, _ = df2
    similar_items = []
    if ref_id in list(feats1[0]):
        sims1 = get_topk_similar(ref_id, df1, distance_str, gen_id_to_account, k=None, verbose=verbose)
        if normalize:
            sims1["dist"] /= np.abs(sims1["dist"].min())
        similar_items.append(sims1)
    if ref_id in list(feats2[0]):
        sims2 = get_topk_similar(ref_id, df2, distance_str, gen_id_to_account, k=None, verbose=verbose)
        if normalize:
            sims2["dist"] /= np.abs(sims2["dist"].min())
        similar_items.append(sims2)
    if len(similar_items) == 2: # combined toplist
        similar_items[0]['dist'] *= alpha
        similar_items[1]['dist'] *= (1.0-alpha)
        both = pd.concat(similar_items)
        sims = both.groupby("id")["dist"].mean().reset_index()
        meta = both[["id","label"]].drop_duplicates()
        sims = sims.merge(meta, on=["id"]).sort_values("dist").reset_index()
    elif len(similar_items) == 1: # single toplist
        sims = similar_items[0]
    else:
        sims = pd.DataFrame([])
    return sims.head(k)

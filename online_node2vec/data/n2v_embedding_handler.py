import pandas as pd

def load_n2v_features(features_dir, delta_time, total_days, player_labels, eval_window=21600, sep=",", verbose=True):
    """Function to load Node2Vec embeddings"""
    offset = eval_window // delta_time
    snapshot_per_day = 86400 // delta_time
    num_all_shapshots = total_days*snapshot_per_day
    first_snapshot = offset - 1
    snapshot_per_day, num_all_shapshots, offset, first_snapshot
    data = []
    print(first_snapshot, num_all_shapshots, offset)
    for i in list(range(first_snapshot,num_all_shapshots,offset)):
        if verbose:
            print(i)
        fname = "%s/embedding_%i.csv" % (features_dir,i)
        feats = pd.read_csv(fname, header=None, sep=sep)
        if verbose:
            print(fname)
        day_idx = i // snapshot_per_day
        label_dict = dict(zip(player_labels[day_idx]["id"],player_labels[day_idx]["label"]))
        data.append((i, feats, label_dict))
    return data

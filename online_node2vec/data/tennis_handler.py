import pandas as pd
import json, os
import twittertennis.handler as tt

def download_data_set(data_dir, data_id):
    handler = tt.TennisDataHandler(data_dir, data_id, include_qualifiers=True)
    print(handler.summary())
    output_dir = "%s/%s_preprocessed" % (data_dir, data_id)
    handler.export_edges(output_dir)
    handler.export_relevance_labels(output_dir, binary=True, only_pos_label=True)
    id_to_account = pd.DataFrame(list(zip(handler.account_to_id.values(), handler.account_to_id.keys())), columns=["id","account"])
    id_to_account.to_csv("%s/id2account.csv" % output_dir, index=False)
    return output_dir, handler

def get_data_info(data_folder, verbose=False):
    if "rg17" in data_folder:
        label_file_range = range(4,19) # files: 4-18: May 28 - June 11 (15-15 day - OK)
    elif "uo17" in data_folder:
        label_file_range = range(7,21) # files: 7-20: Aug 28 - Sept 10
    else:
        raise RuntimeError("data_folder must contain 'rg17' or 'uo17' data identifers!")
    return load_data(data_folder, label_file_range, verbose=verbose)
                
def load_data(folder, label_file_range, verbose=False):
    if verbose:
        print("### Load id to account mapping ###")
    node_id_with_account = pd.read_csv("%s/id2account.csv" % folder)
    gen_id_to_account = dict(zip(node_id_with_account["id"],node_id_with_account["account"]))
    if verbose:
        print("\n### Load daily player labels ###")
    player_labels = []
    for i in label_file_range:
        if verbose:
            print(i)
        tmp_df = pd.read_csv("%s/labels_%i.csv" % (folder,i), sep=" ", names=["id","label"])
        player_labels.append(tmp_df)
    return gen_id_to_account, player_labels

def load_edge_data(data_dir, data_id, max_days=None, sep="|", cols=["time","src","trg"]):
    fp = os.path.join(data_dir, "%s_preprocessed" % data_id, "edges.csv")
    edge_data = pd.read_csv(fp, sep=sep, names=cols)
    if data_id == "rg17":
        # 2017-05-28 0:00 Paris
        start_time = 1495922400
        total_days = 15 if max_days == None else min(15, max_days)
    elif data_id == "uo17":
        # 2017-08-28 0:00 NY
        start_time = 1503892800
        total_days = 14 if max_days == None else min(14, max_days)
    else:
        raise RuntimeError("Invalid dataset!")
    end_time = start_time + total_days*86400
    return edge_data, start_time, end_time
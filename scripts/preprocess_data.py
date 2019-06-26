import sys
import pandas as pd
import twittertennis.handler as tt

if __name__ == "__main__":
    if len(sys.argv) == 2:
        data_dir = sys.argv[1]
        for data_id in ["rg17","uo17"]:
            handler = tt.TennisDataHandler(data_dir, data_id, include_qualifiers=True)
            print(handler.summary())
            output_dir = "%s/%s_preprocessed" % (data_dir, data_id)
            handler.export_edges(output_dir)
            handler.export_relevance_labels(output_dir, binary=True)
            id_to_account = pd.DataFrame(list(zip(handler.account_to_id.values(),handler.account_to_id.keys())),columns=["id","account"])
            id_to_account.to_csv("%s/id2account.csv" % output_dir, index=False)
    else:
        print("Usage: preprocess_data.py <data_dir>")
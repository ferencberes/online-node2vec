import sys, os
from online_node2vec.data.tennis_handler import download_data_set

if __name__ == "__main__":
    if len(sys.argv) == 2:
        data_dir = sys.argv[1]
    else:
        dirpath = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(dirpath, "..", "data")
    for data_id in ["rg17","uo17"]:
        download_data_set(data_dir, data_id)
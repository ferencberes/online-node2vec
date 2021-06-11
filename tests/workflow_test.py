from online_node2vec.data.tennis_handler import download_data_set
import os

dirpath = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dirpath, "..", "data")

def test_data_rg17():
    processed_dir = download_data_set(data_dir, "rg17")
    assert os.path.exists(processed_dir)
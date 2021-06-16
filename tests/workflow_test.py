from online_node2vec.data.tennis_handler import *
from online_node2vec.online.w2v_learners import OnlineWord2Vec
from online_node2vec.online.walk_sampling import StreamWalkUpdater
from online_node2vec.online.online_node2vec_models import LazyNode2Vec
from online_node2vec.online import hash_utils as hu
from online_node2vec.online.walk_sampling import SecondOrderUpdater
import os, shutil

dirpath = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dirpath, "..", "data")
test_folder = "test_results"

def test_data_rg17():
    processed_dir = download_data_set(data_dir, "rg17")
    assert os.path.exists(processed_dir)
    
def test_streamwalk():
    is_fw = False
    edge_data, start_time, end_time = load_edge_data(data_dir, "rg17", 2)
    updater = StreamWalkUpdater(half_life=7200, max_len=2, beta=0.9, cutoff=604800, k=4, full_walks=is_fw)
    learner = OnlineWord2Vec(embedding_dims=128, loss="square", lr_rate=0.035, neg_rate=10, mirror=False, onlymirror=False, init="uniform", exportW1=False, window=2, interval=86400, temporal_noise=False, use_pairs=(not is_fw))
    online_n2v = LazyNode2Vec(updater, learner, is_decayed=True)
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    os.makedirs(test_folder)
    online_n2v.run(edge_data, 6*3600, test_folder, start_time=start_time, end_time=end_time)
    output_dir = os.path.join(test_folder, "lazy_decayedTrue-streamwalk_hl7200_ml2_beta0.90_cutoff604800_k4_fullwFalse-onlinew2v_dim128_lr0.0350_neg10_uratio1.00_square_mirrorFalse_omFalse_inituniform_expW1False_i86400_tnFalse_win2_pairsTrue")
    assert len(os.listdir(output_dir)) == 8
    
def test_second_order():
    edge_data, start_time, end_time = load_edge_data(data_dir, "uo17", 1)
    hash_gen = hu.ModHashGenerator()
    updater = SecondOrderUpdater(half_life=43200, num_hash=20, hash_generator=hash_gen, in_edges=0.0, out_edges=1.0, incr_condition=True)
    learner = OnlineWord2Vec(embedding_dims=128, loss="square", lr_rate=0.01, neg_rate=5, window=0, interval=86400, temporal_noise=False, use_pairs=True, uniform_ratio=0.8)
    # initialize node2vec object
    online_n2v = LazyNode2Vec(updater, learner, True)
    online_n2v.run(edge_data, 6*3600, test_folder, start_time=start_time, end_time=end_time)
    output_dir = os.path.join(test_folder, "lazy_decayedTrue-secondorder_hl43200_numh20_modhash200000_in0.00_out1.00_incrTrue-onlinew2v_dim128_lr0.0100_neg5_uratio0.80_square_mirrorTrue_omFalse_initgensim_expW1True_i86400_tnFalse_win0_pairsTrue")
    assert len(os.listdir(output_dir)) == 4
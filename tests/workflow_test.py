from online_node2vec.data.tennis_handler import *
from online_node2vec.offline.offline_node2vec_model import BatchNode2Vec
from online_node2vec.online.w2v_learners import OnlineWord2Vec, GensimWord2Vec
from online_node2vec.online.walk_sampling import StreamWalkUpdater
from online_node2vec.online.online_node2vec_models import LazyNode2Vec, OnlineNode2Vec
from online_node2vec.online import hash_utils as hu
from online_node2vec.online.walk_sampling import SecondOrderUpdater
import os, shutil

dirpath = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dirpath, "..", "data")
test_folder = "test_results"

def test_data_rg17():
    rg17_dir = download_data_set(data_dir, "rg17")
    uo17_dir = download_data_set(data_dir, "uo17")
    assert os.path.exists(rg17_dir) and os.path.exists(uo17_dir)
    
def test_lazy_streamwalk_with_onlinew2v():
    is_fw = False
    edge_data, start_time, end_time = load_edge_data(data_dir, "rg17", 2)
    updater = StreamWalkUpdater(half_life=7200, max_len=2, beta=0.9, cutoff=604800, k=4, full_walks=is_fw)
    learner = OnlineWord2Vec(embedding_dims=128, loss="sigmoid", lr_rate=0.035, neg_rate=10, mirror=False, onlymirror=False, init="uniform", exportW1=False, window=2, interval=86400, temporal_noise=False, use_pairs=(not is_fw))
    online_n2v = LazyNode2Vec(updater, learner, is_decayed=True)
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    os.makedirs(test_folder)
    online_n2v.run(edge_data, 6*3600, test_folder, start_time=start_time, end_time=end_time)
    output_dir = os.path.join(test_folder, "lazy_decayedTrue-streamwalk_hl7200_ml2_beta0.90_cutoff604800_k4_fullwFalse-onlinew2v_dim128_lr0.0350_neg10_uratio1.00_sigmoid_mirrorFalse_omFalse_inituniform_expW1False_i86400_tnFalse_win2_pairsTrue")
    assert len(os.listdir(output_dir)) == 8
    
def test_online_streamwalk_with_onlinew2v():
    is_fw = False
    edge_data, start_time, end_time = load_edge_data(data_dir, "rg17", 1)
    updater = StreamWalkUpdater(half_life=7200, max_len=2, beta=0.9, cutoff=604800, k=4, full_walks=is_fw)
    learner = OnlineWord2Vec(embedding_dims=128, loss="sigmoid", lr_rate=0.035, neg_rate=10, mirror=False, onlymirror=False, init="uniform", exportW1=False, window=2, interval=86400, temporal_noise=False, use_pairs=(not is_fw))
    online_n2v = OnlineNode2Vec(updater, learner, is_decayed=True)
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    os.makedirs(test_folder)
    online_n2v.run(edge_data, 6*3600, test_folder, start_time=start_time, end_time=end_time)
    output_dir = os.path.join(test_folder, "online_decayedTrue-streamwalk_hl7200_ml2_beta0.90_cutoff604800_k4_fullwFalse-onlinew2v_dim128_lr0.0350_neg10_uratio1.00_sigmoid_mirrorFalse_omFalse_inituniform_expW1False_i86400_tnFalse_win2_pairsTrue")
    assert len(os.listdir(output_dir)) == 4
    
def test_lazy_streamwalk_with_gensimw2v():
    is_fw = False
    edge_data, start_time, end_time = load_edge_data(data_dir, "rg17", 1)
    updater = StreamWalkUpdater(half_life=7200, max_len=2, beta=0.9, cutoff=604800, k=4, full_walks=is_fw)
    learner = GensimWord2Vec(embedding_dims=128, lr_rate=0.01, neg_rate=10, n_threads=1)
    online_n2v = LazyNode2Vec(updater, learner, is_decayed=True)
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    os.makedirs(test_folder)
    online_n2v.run(edge_data, 12*3600, test_folder, start_time=start_time, end_time=end_time)
    output_dir = os.path.join(test_folder, "lazy_decayedTrue-streamwalk_hl7200_ml2_beta0.90_cutoff604800_k4_fullwFalse-gensimw2v_dim128_lr0.0100_neg10_sg1")
    assert len(os.listdir(output_dir)) == 2
    
def test_second_order():
    edge_data, start_time, end_time = load_edge_data(data_dir, "uo17", 1)
    hash_gen = hu.ModHashGenerator()
    updater = SecondOrderUpdater(half_life=43200, num_hash=20, hash_generator=hash_gen, in_edges=0.0, out_edges=1.0, incr_condition=True)
    learner = OnlineWord2Vec(embedding_dims=128, loss="square", lr_rate=0.01, neg_rate=5, window=0, interval=86400, temporal_noise=False, use_pairs=True, uniform_ratio=0.8)
    online_n2v = LazyNode2Vec(updater, learner, True)
    online_n2v.run(edge_data, 6*3600, test_folder, start_time=start_time, end_time=end_time)
    output_dir = os.path.join(test_folder, "lazy_decayedTrue-secondorder_hl43200_numh20_modhash200000_in0.00_out1.00_incrTrue-onlinew2v_dim128_lr0.0100_neg5_uratio0.80_square_mirrorTrue_omFalse_initgensim_expW1True_i86400_tnFalse_win0_pairsTrue")
    assert len(os.listdir(output_dir)) == 4

def test_offline_node2vec():
    edge_data, start_time, end_time = load_edge_data(data_dir, "uo17", 1)
    offline_n2v = BatchNode2Vec(dimensions=128, walk_length=3, num_walks=20, window_size=3, p=1.0, q=1.0, lookback_time=172800, n_threads=1)
    offline_n2v.run(edge_data, 2*3600, test_folder, start_time=start_time, end_time=end_time)
    output_dir = os.path.join(test_folder, "offline_wnum20_wlength3_win3_p1.00_q1.00_dim128_lb172800_dirTrue")
    assert len(os.listdir(output_dir)) == 12

def test_map_hash():
    maph = hu.MapHashGenerator()
    generators = maph.generate(10)
    assert len(generators) == 10
    
def test_mul_hash():
    mulh = hu.MulHashGenerator()
    generators = mulh.generate(2)
    value = 3
    assert generators[0](value) != generators[1](value)
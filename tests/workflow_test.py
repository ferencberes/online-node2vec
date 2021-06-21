from online_node2vec.data.tennis_handler import *
from online_node2vec.data.n2v_embedding_handler import load_n2v_features
from online_node2vec.offline.offline_node2vec_model import BatchNode2Vec
from online_node2vec.online.w2v_learners import OnlineWord2Vec, GensimWord2Vec
from online_node2vec.online.walk_sampling import StreamWalkUpdater
from online_node2vec.online.online_node2vec_models import LazyNode2Vec, OnlineNode2Vec
from online_node2vec.online import hash_utils as hu
from online_node2vec.online.walk_sampling import SecondOrderUpdater
import online_node2vec.evaluation.ndcg_computer as ndcgc
import online_node2vec.evaluation.distance_computer as distc
import online_node2vec.data.n2v_embedding_handler as n2veh
import os, shutil

dirpath = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dirpath, "..", "data")
test_folder = "test_results"

if os.path.exists(test_folder):
    shutil.rmtree(test_folder)
os.makedirs(test_folder)

def test_data_preparation():
    if os.path.exists():
        shutil.rmtree(data_dir)
    rg17_dir = download_data_set(data_dir, "rg17")
    uo17_dir = download_data_set(data_dir, "uo17")
    assert os.path.exists(rg17_dir) and os.path.exists(uo17_dir)
    
def test_lazy_streamwalk_with_onlinew2v():
    is_fw = False
    data_id = "rg17"
    delta_time = 6*3600
    edge_data, start_time, end_time = load_edge_data(data_dir, data_id, 2)
    updater = StreamWalkUpdater(half_life=7200, max_len=2, beta=0.9, cutoff=604800, k=4, full_walks=is_fw)
    learner = OnlineWord2Vec(embedding_dims=128, loss="square", lr_rate=0.035, neg_rate=10, mirror=False, onlymirror=False, init="uniform", exportW1=False, window=2, interval=86400, temporal_noise=False, use_pairs=(not is_fw))
    online_n2v = LazyNode2Vec(updater, learner, is_decayed=True)
    root_dir = "%s/%s/features_%s/delta_%i" % (test_folder, data_id, 0, delta_time)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    output_dir = online_n2v.run(edge_data, delta_time, root_dir, start_time=start_time, end_time=end_time)
    assert len(os.listdir(output_dir)) == 8
    
def test_online_streamwalk_with_onlinew2v():
    is_fw = False
    data_id = "rg17"
    delta_time = 6*3600
    edge_data, start_time, end_time = load_edge_data(data_dir, data_id, 1)
    updater = StreamWalkUpdater(half_life=7200, max_len=2, beta=0.9, cutoff=604800, k=4, full_walks=is_fw)
    learner = OnlineWord2Vec(embedding_dims=128, loss="logsigmoid", lr_rate=0.035, neg_rate=10, mirror=False, onlymirror=False, init="uniform", exportW1=False, window=2, interval=86400, temporal_noise=False, use_pairs=(not is_fw))
    online_n2v = OnlineNode2Vec(updater, learner, is_decayed=True)
    root_dir = "%s/%s/features_%s/delta_%i" % (test_folder, data_id, 0, delta_time)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    output_dir = online_n2v.run(edge_data, delta_time, root_dir, start_time=start_time, end_time=end_time)
    assert len(os.listdir(output_dir)) == 4
    
def test_lazy_streamwalk_with_gensimw2v():
    is_fw = False
    data_id = "uo17"
    delta_time = 12*3600
    edge_data, start_time, end_time = load_edge_data(data_dir, data_id, 1)
    updater = StreamWalkUpdater(half_life=7200, max_len=2, beta=0.9, cutoff=604800, k=4, full_walks=is_fw)
    learner = GensimWord2Vec(embedding_dims=128, lr_rate=0.01, neg_rate=10, n_threads=1)
    online_n2v = LazyNode2Vec(updater, learner, is_decayed=True)
    root_dir = "%s/%s/features_%s/delta_%i" % (test_folder, data_id, 0, delta_time)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    output_dir = online_n2v.run(edge_data, delta_time, root_dir, start_time=start_time, end_time=end_time)
    assert len(os.listdir(output_dir)) == 2
    
def test_second_order():
    data_id = "rg17"
    delta_time = 6*3600
    edge_data, start_time, end_time = load_edge_data(data_dir, data_id, 1)
    hash_gen = hu.ModHashGenerator()
    updater = SecondOrderUpdater(half_life=43200, num_hash=20, hash_generator=hash_gen, in_edges=0.0, out_edges=1.0, incr_condition=True)
    learner = OnlineWord2Vec(embedding_dims=128, loss="square", lr_rate=0.01, neg_rate=5, window=0, interval=86400, temporal_noise=False, use_pairs=True, uniform_ratio=0.8)
    online_n2v = LazyNode2Vec(updater, learner, True)
    root_dir = "%s/%s/features_%s/delta_%i" % (test_folder, data_id, 0, delta_time)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    output_dir = online_n2v.run(edge_data, delta_time, root_dir, start_time=start_time, end_time=end_time)
    assert len(os.listdir(output_dir)) == 4
    
def test_offline_node2vec():
    edge_data, start_time, end_time = load_edge_data(data_dir, "uo17", 1)
    offline_n2v = BatchNode2Vec(dimensions=128, walk_length=3, num_walks=20, window_size=3, p=1.0, q=1.0, lookback_time=172800, n_threads=1)
    output_dir = offline_n2v.run(edge_data, 2*3600, test_folder, start_time=start_time, end_time=end_time)
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

def test_evaluation():
    data_id = "rg17"
    delta_time = 6*3600
    total_days = 2
    sample_id = 0
    parameters = "lazy_decayedTrue-streamwalk_hl7200_ml2_beta0.90_cutoff604800_k4_fullwFalse-onlinew2v_dim128_lr0.0350_neg10_uratio1.00_square_mirrorFalse_omFalse_inituniform_expW1False_i86400_tnFalse_win2_pairsTrue"
    root_dir_prefix = "%s/%s/features_%s/delta_%i" % (test_folder, data_id, sample_id, delta_time)
    ndcg_eval_dir = "%s/%s/eval_%i/delta_%i/" % (test_folder, data_id, sample_id, delta_time)
    ndcg_eval_file = "%s/%s.csv" % (ndcg_eval_dir, parameters)
    features_dir = "%s/%s/features_%i/delta_%i/%s" % (test_folder, data_id, sample_id, delta_time, parameters)
    gen_id_to_account, player_labels = get_data_info(os.path.join(data_dir, "%s_preprocessed" % data_id))
    data = load_n2v_features(features_dir, delta_time, total_days, player_labels, delta_time, sep=",")
    res_dot = ndcgc.parallel_eval_ndcg(data, gen_id_to_account, "-dot", n_threads=2)  
    all_df = pd.concat(res_dot)
    if not os.path.exists(ndcg_eval_dir):
        os.makedirs(ndcg_eval_dir)
    all_df.to_csv(ndcg_eval_file)
    mean_ndcg = all_df["ndcg"].mean()
    assert 0 <= mean_ndcg and mean_ndcg <= 1.0

def test_eval_combination():
    max_threads = 2
    data_id = "rg17"
    total_days = 1
    delta_time = 21600
    ## Load label information
    gen_id_to_account, player_labels = get_data_info(os.path.join(data_dir, "%s_preprocessed" % data_id))
    ## Load embeddings
    model_dirs = {
            "so" : "%s/rg17/features_0/delta_21600/lazy_decayedTrue-secondorder_hl43200_numh20_modhash200000_in0.00_out1.00_incrTrue-onlinew2v_dim128_lr0.0100_neg5_uratio0.80_square_mirrorTrue_omFalse_initgensim_expW1True_i86400_tnFalse_win0_pairsTrue/" % test_folder,
            "sw" : "%s/rg17/features_0/delta_21600/lazy_decayedTrue-streamwalk_hl7200_ml2_beta0.90_cutoff604800_k4_fullwFalse-onlinew2v_dim128_lr0.0350_neg10_uratio1.00_square_mirrorFalse_omFalse_inituniform_expW1False_i86400_tnFalse_win2_pairsTrue/" % test_folder
        }
    feature_sets = {}
    feature_sets["so"] = n2veh.load_n2v_features(model_dirs["so"], delta_time, total_days, player_labels, verbose=False)
    feature_sets["sw"] = n2veh.load_n2v_features(model_dirs["sw"], delta_time, total_days, player_labels, verbose=False)
    print("done")
    # 1. Standalone performance
    second_order_res = ndcgc.parallel_eval_ndcg(feature_sets["so"], gen_id_to_account, "-dot", n_threads=max_threads)
    first_order_res = ndcgc.parallel_eval_ndcg(feature_sets["sw"], gen_id_to_account, "-dot", n_threads=max_threads)
    so_perf = pd.concat(second_order_res)["ndcg"].mean()
    fo_perf = pd.concat(first_order_res)["ndcg"].mean()
    print("SO:", so_perf)
    print("FO:", fo_perf)
    # 2. SO+SW (weighted) combination
    so_weight = 0.3
    combi_res = ndcgc.parallel_combined_eval_ndcg(feature_sets["so"], feature_sets["sw"], so_weight, gen_id_to_account, "-dot", n_threads=max_threads)
    combi_perf = pd.concat(combi_res)["ndcg"].mean()
    print("Combi:", combi_perf)
    assert 0 <= combi_perf and combi_perf <= 1.0

def test_toplist_combination():
    data_id = "rg17"
    total_days = 1
    gen_id_to_account, player_labels = get_data_info("%s/%s_preprocessed" % (data_dir, data_id))
    delta_time = 21600
    feature_sets = {}
    model_dirs = {
            "so" : "%s/rg17/features_0/delta_21600/lazy_decayedTrue-secondorder_hl43200_numh20_modhash200000_in0.00_out1.00_incrTrue-onlinew2v_dim128_lr0.0100_neg5_uratio0.80_square_mirrorTrue_omFalse_initgensim_expW1True_i86400_tnFalse_win0_pairsTrue/" % test_folder,
            "sw" : "%s/rg17/features_0/delta_21600/lazy_decayedTrue-streamwalk_hl7200_ml2_beta0.90_cutoff604800_k4_fullwFalse-onlinew2v_dim128_lr0.0350_neg10_uratio1.00_square_mirrorFalse_omFalse_inituniform_expW1False_i86400_tnFalse_win2_pairsTrue/" % test_folder
        }
    feature_sets["so"] = n2veh.load_n2v_features(model_dirs["so"], delta_time, total_days, player_labels, verbose=False)
    feature_sets["sw"] = n2veh.load_n2v_features(model_dirs["sw"], delta_time, total_days, player_labels, verbose=False)
    #toplist combination
    snapshot_idx = 1
    ref_id = 14571755
    for metric in ["-dot","euclidean","cosine","1-pearson"]:
        combi_res = distc.get_combined_topk_similar(ref_id, 0.3, feature_sets["so"][snapshot_idx], feature_sets["sw"][snapshot_idx], metric, gen_id_to_account, k=20, normalize=True, verbose=True)
        assert len(combi_res) == 20
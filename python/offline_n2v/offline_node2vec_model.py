import os, sys
import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
from .node2vec import Graph

sys.path.insert(0,"../")
from online_n2v.online_node2vec_models import Node2VecBase

class BatchNode2Vec(Node2VecBase):
    def __init__(self, dimensions=128, walk_length=5, num_walks=10, window_size=3, p=1.0, q=1.0, lookback_time=12*3600, directed=True, num_iters=1, n_threads=4):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.p = p
        self.q = q
        self.lookback_time = lookback_time
        self.directed = directed
        self.workers = n_threads
        self.iter = num_iters
        super(BatchNode2Vec, self).__init__(None, None, False, False)
        
    def __str__(self):
        return "offline_wnum%i_wlength%i_win%i_p%.2f_q%.2f_dim%i_lb%i_dir%s" % (self.num_walks, self.walk_length, self.window_size, self.p, self.q, self.dimensions, self.lookback_time, self.directed)
    
    def get_embeddings(self, model):
        vectors = model.wv.vectors
        embeddings = pd.DataFrame(vectors).reset_index()
        embeddings['index'] = model.wv.index2word
        return embeddings
    
    def learn_embeddings(self, walks):
        '''Learn embeddings by optimizing the Skipgram objective using SGD.'''
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(walks, size=self.dimensions, window=self.window_size, min_count=0, sg=1, workers=self.workers, iter=self.iter)
        #model.wv.save_word2vec_format(args.output)
        return self.get_embeddings(model)

    def train(self, nx_G):
        '''Pipeline for representational learning for all nodes in a graph.'''
        G = Graph(nx_G, self.directed, self.p, self.q)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(self.num_walks, self.walk_length)
        return self.learn_embeddings(walks)

    def run(self, edge_data, snapshot_window, output_dir, start_time, end_time=None):
        """Edges have to be sorted according to time column."""
        # filter data (global filter)
        partial_data = super(BatchNode2Vec, self).filter_edges(edge_data, start_time, end_time)
        num_snapshots = (partial_data["time"].max() - start_time) // snapshot_window + 1
        experiment_dir = "%s/%s/" % (output_dir, str(self))
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        for idx in range(1, num_snapshots+1):
            snapshot_epoch = start_time + idx * snapshot_window
            snapshot_data = super(BatchNode2Vec, self).filter_edges(partial_data, snapshot_epoch-self.lookback_time, snapshot_epoch, verbose=False)
            print("snapshot %i" % idx, "num_edges %i" % len(snapshot_data))
            # only unweighted case is implemented
            snapshot_data['weight'] = 1
            G = nx.from_pandas_edgelist(snapshot_data, 'src', 'trg', edge_attr=True, create_using=nx.DiGraph())
            if not self.directed:
                G = G.to_undirected()
            embeddings = self.train(G)
            output_name = "%s/embeddings_%i.csv" % (experiment_dir, idx)
            embeddings.to_csv(output_name, index=False, header=False)
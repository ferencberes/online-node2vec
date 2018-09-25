import time, os
import pandas as pd
import numpy as np
from .w2v_learners import OnlineWord2Vec
from .walk_sampling import *

class OnlineNode2Vec():
    def __init__(self, is_directed, logging):
        """Abstract class for Online Node2Vec"""
        self.is_directed = is_directed
        self.logging = logging
        self.learner = None
        self.node_last_update = {}

    def filter_edges(self, edge_df, start_time, end_time):
        """Filter edges based on time. Drop loop edges."""
        print("Original number of edges: %i" % len(edge_df))
        if start_time == None:
            start_time = int(edge_df["time"].min())
        if end_time == None:
            end_time = int(edge_df["time"].max())
        partial_data = edge_df[(edge_df["time"] >= start_time) & (edge_df["time"] < end_time)]
        print("Number of edges after temporal filter: %i" % len(partial_data))
        partial_data = partial_data[partial_data["src"] != partial_data["trg"]]
        print("Number of edges after dropping loop edges: %i" % len(partial_data))
        nodes = set(partial_data["src"]).union(partial_data["trg"])
        print("Number of nodes in the remaining data: %i" % len(nodes))
        nodes_str = [str(n) for n in nodes]
        return partial_data, nodes_str
        
    def lazy_train_model(self):
        """Lazy model training for multiple node pairs"""
        #print(len(self.sampled_pairs))
        if len(self.sampled_pairs) > 0 and self.learner != None:
            train_time_start = time.time()
            #print(self.sampled_pairs)
            self.learner.partial_fit(self.sampled_pairs)
            train_time_stop = time.time()
            self.sum_train_time += (train_time_stop - train_time_start)
            self.sampled_pairs = []
        
    def run_base(self, partial_data, snapshot_window, output_dir, start_time):
        start_epoch = int(time.time())
        last_snapshot_epoch, snapshot_idx = start_time, 0
        self.sum_train_time = 0.0
        self.sampled_pairs = []
        print("Experiment was STARTED")
        for edge_num, row in partial_data.iterrows():
            current_time, source, target = row["time"], row["src"], row["trg"]
            # synchorization at snapshot barriers
            if current_time-last_snapshot_epoch > snapshot_window:
                # lazy learning
                self.lazy_train_model()
                # export embedding
                self.export_features(output_dir, snapshot_idx, start_epoch, last_snapshot_epoch+snapshot_window)
                last_snapshot_epoch += snapshot_window
                snapshot_idx += 1
            self.node_last_update[source] = current_time
            self.node_last_update[target] = current_time
            # update & sample node pairs for model training
            new_pairs = self.updater.process_new_edge(source, target, current_time)
            if not self.is_directed:
                # handle edges like undirected
                more_pairs = self.updater.process_new_edge(target, source, current_time)
                new_pairs += more_pairs
            self.sampled_pairs += new_pairs
        # lazy learning
        self.lazy_train_model()
        # export embedding
        self.export_features(output_dir, snapshot_idx, start_epoch, current_time)
        print("Experiment was FINISHED")

class CombinatedMethod(OnlineNode2Vec):
    def __init__(self, name, tempWalk_path, secOrder_path, percentage, snapshot_num, dimension=128, lr_rate=0.01, sg=1, neg_rate=5, half_life=3600*11, is_decayed=False, restart_rate=None, overlap_size=None, n_threads=1, logging=False, just_second=False, online_w2v_model=False):
        """Online Node2Vec implementation with updated temporal weighted walks"""
        self.restart_rate = restart_rate
        self.overlap_size = overlap_size
        self.dimension = dimension
        self.lr_rate = lr_rate
        self.sg = sg
        self.neg_rate = neg_rate
        self.n_threads = n_threads
        self.snapshot_num = snapshot_num
        self.tempWalk_path = tempWalk_path
        self.secOrder_path = secOrder_path
        self.percentage = percentage
        self.just_second = just_second
        self.online_w2v_model = online_w2v_model
        self.is_decayed = is_decayed
        self.half_life = half_life

        if self.restart_rate == None:
            self.restart_rate = snapshot_num+1
            self.overlap_size = 0

        self.model_str = "%s_decayed%s_dim%i_lr%0.4f_sg%i_neg%i_mO%s_resRate%s_ovSize%s" % (name, str(self.is_decayed), dimension, lr_rate, sg, neg_rate, str(online_w2v_model), str(self.restart_rate), str(self.overlap_size))
        super(CombinatedMethod, self).__init__(is_directed=False, logging=logging)

    def run(self, edge_data, output_dir, start_time=None, end_time=None):
        """Edges have to be sorted according to time column"""
        partial_data, nodes_str = self.filter_edges(edge_data, start_time, end_time)
        # online Word2Vec is trained with window=1, and epochs=1 (due to the fact that the sencentes are only node pairs)
        self.learner = OnlineWord2Vec(nodes_str, embedding_dims=self.dimension, window_size=1, num_epochs=1, lr_rate=self.lr_rate, sg=self.sg, neg_rate=self.neg_rate, n_threads=self.n_threads,  online_w2v_model=self.online_w2v_model)

        print("W2V learner was INITIALIZED")
        #super(OnlineNode2VecSecondOrderSim, self).run_base(partial_data, snapshot_window, output_dir, start_time)
        #self.run_base(partial_data, snapshot_window, output_dir, start_time)      

        #read data
        pairs = []
        for period in range(self.snapshot_num):
            temp = pd.read_csv(self.tempWalk_path + 'extended_chosen_df_' + str(period) + '.csv')
            sec  = pd.read_csv(self.secOrder_path  + 'extended_chosen_df_' + str(period) + '.csv')
            
            all_df = pd.concat([temp, sec], ignore_index=True).sort_values(by=['t', 'u']).reset_index(drop=True)
            pairs.append(all_df)

        self.sampled_pairs = []
        count_period = 0
        for period_out in range(0, self.snapshot_num, self.restart_rate):
            
            from_period = period_out - self.overlap_size
            if from_period < 0:
                from_period = 0    

            end_period = period_out+self.restart_rate
            if end_period > self.snapshot_num:
                end_period = self.snapshot_num

            self.learner = OnlineWord2Vec(nodes_str, embedding_dims=self.dimension, window_size=1, num_epochs=1, lr_rate=self.lr_rate, sg=self.sg, neg_rate=self.neg_rate, n_threads=self.n_threads,  online_w2v_model=self.online_w2v_model)

            for period in range(from_period, end_period, 1):   
                
                samples = []

                last_time_point = None
                for timepoint in pairs[period]['t'].unique():
                    splitted_data = pairs[period][pairs[period]['t'] == timepoint]
                    
                    temp_data = splitted_data[splitted_data['method'] == 'tempWalk']
                    seco_data = splitted_data[splitted_data['method'] == 'secOrdSim']
                    
                    if self.just_second == False:
                        for index, row in temp_data.iterrows():
                            source = int(row['s'])
                            target = int(row['w'])
                            samples.append( ( str(source), str( target ) ) )
                            self.node_last_update[source] = timepoint
                            self.node_last_update[target] = timepoint
                    
                    if self.percentage != None:
                        sec_sample_number = int(self.percentage * temp_data.shape[0])
                        if sec_sample_number > seco_data.shape[0]:
                            sec_sample_number = seco_data.shape[0]

                        if sec_sample_number != 0:
                            for index, row in seco_data.sample(n=sec_sample_number).iterrows():
                                source = int(row['s'])
                                target = int(row['w'])
                                samples.append( ( str(source), str( target ) ) )
                                self.node_last_update[source] = timepoint
                                self.node_last_update[target] = timepoint
                    else:
                        for index, row in seco_data.iterrows():
                            source = int(row['s'])
                            target = int(row['w'])
                            samples.append( ( str(source), str( target ) ) )
                            self.node_last_update[source] = timepoint
                            self.node_last_update[target] = timepoint

                    if self.online_w2v_model:
                        self.learner.partial_fit(samples, interval=150)
                
                    last_time_point = timepoint

                start_epoch = int(time.time())
                if not self.online_w2v_model:
                    self.learner.partial_fit(samples)

                if period >= period_out:
                    self.export_features(output_dir, period, start_time, last_time_point)
                    print('period ', period, ': ', len(samples), ' sample')

                self.sampled_pairs.append(samples)



    def export_features(self, output_dir, snapshot_idx, start_epoch, snapshot_time=None):
        """Export features if 'output_dir' was specified. Also prints runing time information."""
        elapsed_seconds = int(time.time())-start_epoch
        if output_dir != None:
            model_out_dir = "%s/%s" % (output_dir, self.model_str)
            if not os.path.exists(model_out_dir):
                os.makedirs(model_out_dir)
            file_name = "%s/embedding_%i.csv" % (model_out_dir, snapshot_idx)

            if self.is_decayed:
                # apply decay on embeddings
                now = snapshot_time
                c = - np.log(0.5) / self.half_life
                embeddings = self.learner.model.wv.vectors
                N, dim = embeddings.shape
                decays = []
                idx2words = self.learner.model.wv.index2word
                for idx, word in enumerate(idx2words):
                    decay = np.exp(-c*(now-self.node_last_update[int(word)]))
                    decays.append(decay)
                decays = np.array(decays).reshape(N,1)
                decayed_embeddings = embeddings * decays
                cols = list(range(1,dim+1))
                # export decayed embeddings
                output_df = pd.DataFrame(decayed_embeddings, columns=cols)
                output_df["id"] = idx2words
                cols = ["id"] + cols
                output_df[cols].to_csv(file_name, index=False)
            else:
                self.learner.export_embeddings(file_name)
        else:
            print("'output_dir' was not specified. Embedding was not exported!")
        print(snapshot_idx, elapsed_seconds)

import time, os
import pandas as pd
import numpy as np
from .w2v_learners import OnlineWord2Vec
from .walk_sampling import *

class Node2VecBase():
    def __init__(self, updater, learner, is_decayed):
        self.updater = updater
        self.learner = learner
        self.is_decayed = is_decayed
        self.node_last_update = {}
        print("Model was INITIALIZED: %s" % str(self))
        
    def filter_edges(self, edge_df, start_time, end_time, verbose=True):
        """Filter edges based on time. Drop loop edges."""
        if verbose:
            print("Original number of edges: %i" % len(edge_df))
        if start_time == None:
            start_time = int(edge_df["time"].min())
        if end_time == None:
            end_time = int(edge_df["time"].max())
        partial_data = edge_df[(edge_df["time"] >= start_time) & (edge_df["time"] < end_time)]
        if verbose:
            print("Number of edges after temporal filter: %i" % len(partial_data))
        partial_data = partial_data[partial_data["src"] != partial_data["trg"]]
        if verbose:
            print("Number of edges after dropping loop edges: %i" % len(partial_data))
        nodes = set(partial_data["src"]).union(set(partial_data["trg"]))
        if verbose:
            print("Number of nodes in the remaining data: %i" % len(nodes))
        nodes_str = [str(n) for n in nodes]
        # setting the total node set as words for the learner
        if self.learner != None:
            self.learner.set_all_words(nodes_str)
        return partial_data
    
    def export_features(self, output_dir, snapshot_idx, start_epoch, snapshot_time):
        """Export features if 'output_dir' was specified. Also prints runing time information."""
        elapsed_seconds = int(time.time())-start_epoch
        if output_dir != None:
            # export only already activated nodes
            activated_nodes = list(self.node_last_update.keys())
            model_out_dir = "%s/%s" % (output_dir, str(self))
            file_name = "%s/embedding_%i.csv" % (model_out_dir, snapshot_idx)
            if not os.path.exists(model_out_dir):
                os.makedirs(model_out_dir)
            if self.is_decayed:
                # apply decay on embeddings
                decay_info = (snapshot_time, self.updater.c, self.node_last_update)
                self.learner.export_embeddings(file_name, nbunch=activated_nodes, decay_information=decay_info)
            else:
                self.learner.export_embeddings(file_name, nbunch=activated_nodes, decay_information=None)
        else:
            print("'output_dir' was not specified. Embedding was not exported!")
        if "TemporalWalk" in str(type(self.updater)):
            print(snapshot_idx, elapsed_seconds, self.sum_train_time, self.updater.num_stored_walks)
        else:
            print(snapshot_idx, elapsed_seconds, self.sum_train_time)
        return model_out_dir
        
class LazyNode2Vec(Node2VecBase):
    """
    Object for running online node embedding algorithms. `LazyNode2Vec` only updates node representations before exporting them for each snapshot. The delayed representation update improves running time compared to the `OnlineNode2Vec` implementation.
    
    Parameters
    ----------
    updater : object
        Specify an Updater object that samples node pairs
    learner : object
        Specify a Word2Vec object to learn online node representations 
    is_decayed: bool
        Apply time decay for the exported node representations. The representation of recently inactive nodes will be close to the origo.
    """
    def __init__(self, updater, learner, is_decayed=False):
        super(LazyNode2Vec, self).__init__(updater, learner, is_decayed)
        
    def __str__(self):
        return "lazy_decayed%s-%s-%s" % (self.is_decayed, self.updater, self.learner)

    def lazy_train_model(self, current_time):
        """Lazy model training for multiple node pairs"""
        #print(len(self.sampled_pairs))
        if len(self.sampled_pairs) > 0 and self.learner != None:
            train_time_start = time.time()
            #print(self.sampled_pairs)
            self.learner.partial_fit(self.sampled_pairs, current_time)
            train_time_stop = time.time()
            self.sum_train_time += (train_time_stop - train_time_start)
            self.sampled_pairs = []

    def run(self, edge_data, snapshot_window, output_dir, start_time, end_time=None):
        """
        Run online node2vec experiment. Node representation are exported for each snapshot.

        Parameters
        ----------
        edge_data : 
            Edge stream data. Edges have to be sorted according to time column.
        snapshot_window : int
            Snapshot length in seconds
        output_dir: str
            Folder to export representations. Only the root is needed. Parameters will be appended to a subfolder.
        start_time: int
            Starting epoch of the experiment
        end_time: int
            Ending epoch of the experiment
        """
        # filter data
        partial_data = super(LazyNode2Vec, self).filter_edges(edge_data, start_time, end_time)
        start_epoch = int(time.time())
        last_snapshot_epoch, snapshot_idx = start_time, 0
        self.sum_train_time = 0.0
        self.sampled_pairs = []
        print("Experiment was STARTED")
        for edge_num, row in partial_data.iterrows():
            current_time, source, target = row["time"], str(int(row["src"])), str(int(row["trg"]))
            # synchorization at snapshot barriers
            if current_time-last_snapshot_epoch > snapshot_window:
                # lazy learning
                self.lazy_train_model(current_time)
                # export embedding
                self.export_features(output_dir, snapshot_idx, start_epoch, last_snapshot_epoch+snapshot_window)
                last_snapshot_epoch += snapshot_window
                snapshot_idx += 1
            self.node_last_update[source] = current_time
            self.node_last_update[target] = current_time
            # update & sample node pairs for model training
            new_pairs = self.updater.process_new_edge(source, target, current_time)
            self.sampled_pairs += new_pairs
        # lazy learning
        self.lazy_train_model(current_time)
        # export embedding
        model_out_dir = self.export_features(output_dir, snapshot_idx, start_epoch, current_time)
        print("Experiment was FINISHED")
        return model_out_dir
        
class OnlineNode2Vec(Node2VecBase):
    """
    Object for running online node embedding algorithms. `OnlineNode2Vec` updates node representations after every link arrival in the edge stream.
    
    Parameters
    ----------
    updater : object
        Specify an Updater object that samples node pairs
    learner : object
        Specify a Word2Vec object to learn online node representations 
    is_decayed: bool
        Apply time decay for the exported node representations. The representation of recently inactive nodes will be close to the origo.
    """
    def __init__(self, updater, learner, is_decayed=False):
        super(OnlineNode2Vec, self).__init__(updater, learner, is_decayed)
        
    def __str__(self):
        return "online_decayed%s-%s-%s" % (self.is_decayed, self.updater, self.learner)

    def online_train_model(self, sampled_pairs, current_time):
        """Online model training for multiple node pairs"""
        #print(len(self.sampled_pairs))
        if len(sampled_pairs) > 0 and self.learner != None:
            train_time_start = time.time()
            #print(sampled_pairs)
            self.learner.partial_fit(sampled_pairs, current_time)
            train_time_stop = time.time()
            self.sum_train_time += (train_time_stop - train_time_start)

    def run(self, edge_data, snapshot_window, output_dir, start_time, end_time=None):
        """
        Run online node2vec experiment. Node representation are exported for each snapshot.

        Parameters
        ----------
        edge_data : 
            Edge stream data. Edges have to be sorted according to time column.
        snapshot_window : int
            Snapshot length in seconds
        output_dir: str
            Folder to export representations. Only the root is needed. Parameters will be appended to a subfolder.
        start_time: int
            Starting epoch of the experiment
        end_time: int
            Ending epoch of the experiment
        """
        # filter data
        partial_data = super(OnlineNode2Vec, self).filter_edges(edge_data, start_time, end_time)
        start_epoch = int(time.time())
        last_snapshot_epoch, snapshot_idx = start_time, 0
        self.sum_train_time = 0.0
        print("Experiment was STARTED")
        for edge_num, row in partial_data.iterrows():
            current_time, source, target = row["time"], str(int(row["src"])), str(int(row["trg"]))
            # synchorization at snapshot barriers
            if current_time-last_snapshot_epoch > snapshot_window:
                # export embedding
                self.export_features(output_dir, snapshot_idx, start_epoch, last_snapshot_epoch+snapshot_window)
                last_snapshot_epoch += snapshot_window
                snapshot_idx += 1
            self.node_last_update[source] = current_time
            self.node_last_update[target] = current_time
            # update & sample node pairs for model training
            new_pairs = self.updater.process_new_edge(source, target, current_time)
            self.online_train_model(new_pairs, current_time)
            # update seen graph for npw2v model
            self.learner.add_edge(source, target, current_time)
        # export embedding
        model_out_dir = self.export_features(output_dir, snapshot_idx, start_epoch, current_time)
        print("Experiment was FINISHED")
        return model_out_dir
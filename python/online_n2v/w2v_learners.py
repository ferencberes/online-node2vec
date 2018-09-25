from gensim.models import Word2Vec
from .npw2v import NPWord2Vec
import pandas as pd
import numpy as np

class Word2VecBase():
    def __init__(self):
        self.all_words = None
        self.model = None
    
    def set_all_words(self, all_words):
        self.all_words = all_words
    
    def get_rank(self, src, trg, top_k):
        raise RuntimeError("Implement this method for each subclass!")
    
    def export_embeddings(self, file_name, nbunch=None, decay_information=None):
        """"Export online word2vec features"""
        #print(file_name)
        if self.model == None:
            with open(file_name, 'w') as f:
                f.write("No word2vec model is available! No training instances were recieved.")
        else:
            embeddings = self.get_embeddings()
            if nbunch != None:
                embeddings = embeddings[embeddings['index'].isin(nbunch)]
            if decay_information != None:
                now, c, node_last_update = decay_information
                embeddings = embeddings.set_index('index')
                decays = []
                for node_id in list(embeddings.index):
                    decay = np.exp(-c*(now-node_last_update[node_id]))
                    decays.append(decay)
                decays_reshaped = np.array(decays).reshape(len(decays),1)
                embeddings = decays_reshaped * embeddings
                embeddings = embeddings.reset_index()        
            embeddings.to_csv(file_name, index=False, header=False)

class OnlineWord2Vec(Word2VecBase):
    def __init__(self, embedding_dims=128, lr_rate=0.01, neg_rate=10, interval=3600, temporal_noise=False):
        """Wrapper for Online Word2Vec model implemented by Kelen Domokos."""
        self.embedding_dims = embedding_dims
        self.lr_rate = lr_rate
        self.neg_rate = neg_rate
        self.interval = interval
        self.temporal_noise = temporal_noise
        super(OnlineWord2Vec, self).__init__()

    def __str__(self):
        return "onlinew2v_dim%i_lr%0.4f_neg%i_i%i_tn%s" % (self.embedding_dims, self.lr_rate, self.neg_rate, self.interval, self.temporal_noise)
        
    def partial_fit(self, sentences, time):
        """Note: learning rate is fixed during online training."""
        if self.model == None:
            self.last_update = time
            self.appearences = []
            if self.all_words == None:
                raise RuntimeError("'all_words' must be set before initialization!")
            self.model =  NPWord2Vec(self.all_words, self.embedding_dims, self.lr_rate, self.neg_rate)
        #refresh noise
        for (a, b) in sentences:
            self.appearences += [a,b]
        time_diff = time - self.last_update
        if time_diff > self.interval:
            print("Updating noise with %i records" % len(self.appearences))
            self.model.update_noise_dist(self.appearences)
            self.last_update += self.interval
            if self.temporal_noise:
                self.appearences = []
        # update model
        self.model.train_pairs(sentences)
        
    def get_rank(self, src, trg, top_k):
        if self.model == None:
            return None
        else:
            src_idx = self.model.vocab_code_map[src]
            trg_idx = self.model.vocab_code_map[trg]
            return self.model.get_rank(src_idx, trg_idx, top_k)

    def get_embeddings(self):
        W, vocab_code_map  = self.model.get_embed()
        reverse_map = {v:k for k,v in vocab_code_map.items()}
        embeddings = pd.DataFrame(W).reset_index()
        embeddings['index'] = embeddings['index'].map(reverse_map)
        return embeddings

class GensimWord2Vec(Word2VecBase):
    def __init__(self, embedding_dims=128, lr_rate=0.01, sg=1, neg_rate=10, n_threads=4):
        """Gensim online Word2Vec model wrapper object."""
        self.embedding_dims = embedding_dims
        self.lr_rate = lr_rate
        self.sg = sg
        self.neg_rate = neg_rate
        self.n_threads = n_threads
        self.num_epochs = 1
        self.closest_ids = {}
        self.embeddings = None
        super(GensimWord2Vec, self).__init__()
        
    def get_rank(self, src, trg, topk):
        if self.embeddings != None:
            if src not in self.closest_ids:
                if src in self.embeddings:
                    self.get_closest_ids(src, topk)
                else:
                    return None
            if trg in self.closest_ids[src]:
                rank = self.closest_ids[src].index(trg)
                # min value should be one
                return rank + 1
        return None
    
    def get_closest_ids(self, src, topk):
        src_vec = self.embeddings[src]
        id_list = np.array([idx for idx in self.embeddings.keys() if idx!=src])
        vec_dot = np.array([src_vec.dot(self.embeddings[idx]) for idx in id_list])
        #argsort in descending order, topk values
        needed_id_places = np.argsort(vec_dot)[::-1][:topk]
        #ids
        self.closest_ids[src] = list(id_list[needed_id_places])
        

    def __str__(self):
        return "gensimw2v_dim%i_lr%0.4f_neg%i_sg%i" % (self.embedding_dims, self.lr_rate, self.neg_rate, self.sg)
        
    def partial_fit(self, sentences, time=None):
        """Note: learning rate is fixed during online training. Time parameter is not used!"""
        if self.model == None:
            if self.all_words == None:
                raise RuntimeError("'all_words' must be set before initialization!")
            if self.neg_rate < 0:
                self.model = Word2Vec(sentences, min_count=1, size=self.embedding_dims, window=1, alpha=self.lr_rate, min_alpha=self.lr_rate, sg=self.sg, negative=0, hs=1, iter=self.num_epochs, workers=self.n_threads) #hierarchical softmax
            else:
                self.model = Word2Vec(sentences, min_count=1, size=self.embedding_dims, window=1, alpha=self.lr_rate, min_alpha=self.lr_rate, sg=self.sg, negative=self.neg_rate, iter=self.num_epochs, workers=self.n_threads)
        # update model
        self.model.build_vocab(sentences, update=True)
        self.model.train(sentences, epochs=self.num_epochs, total_words=len(self.all_words))
        self.embeddings = self.get_embedding_vectors()
        self.closest_ids = {}
    
    def get_embedding_vectors(self):
        vectors = self.model.wv.vectors
        indices = self.model.wv.index2word
        embeddings = {indices[i]:vectors[i] for i in range(len(indices))}
        return embeddings

    def get_embeddings(self):
        vectors = self.model.wv.vectors
        embeddings = pd.DataFrame(vectors).reset_index()
        embeddings['index'] = self.model.wv.index2word
        return embeddings     

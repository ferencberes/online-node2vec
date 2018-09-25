import numpy as np
from scipy.special import expit
import pandas as pd

class NPWord2Vec:
    def __init__(self, vocabulary, embedding_dim, learning_rate=0.05, negative_rate = 5, ns_exponent=0.75):
        #vocab
        self.vocabulary = np.unique(list(vocabulary))
        self.vocab_size = len(self.vocabulary)
        self.vocab_codes = np.arange(self.vocab_size)
        self.vocab_code_map = dict(zip(self.vocabulary, self.vocab_codes))

        #params
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.negative_rate = negative_rate
        self.ns_exponent = ns_exponent
        
        #state
        self.W1 = (np.random.rand(self.vocab_size, self.embedding_dim)-0.5)/self.embedding_dim
        self.W2 = np.zeros((self.vocab_size, self.embedding_dim))
        self.noise_dist_counts = np.zeros(self.vocab_size)+0.01

        #etc
        self.vs = np.array_split(self.W1, self.W1.shape[0]//110)
        self.neg_ref = np.zeros(negative_rate+1)
        self.neg_ref[0] = 1
        self.update_noise_dist(self.vocabulary)

    def train_sentence(self, sentence, window=10, learning_rate=None, negative_rate=None):
        l = len(sentence)
        if(l==1):
            return
        pairs = []
        for s1 in range(l):
            reduced_window = np.random.randint(window-1)
            for s2 in range(max(0,s1-window),min(l,s1+window)):
                if(s1!=s2):
                    pairs.append((sentence[s1],sentence[s2]))
        self.train_pairs(pairs, False, learning_rate, negative_rate)

    def train_pairs(self, pairs, mirror=True, learning_rate = None, negative_rate = None):
        for (s1,s2) in pairs:
            self.train_pair(s1,s2,learning_rate,negative_rate)
        if(mirror):
            for (s2,s1) in pairs:
                self.train_pair(s2,s1,learning_rate,negative_rate)

    def train_pair(self, s1, s2,learning_rate=None,negative_rate=None):
        nr = negative_rate if negative_rate is not None else self.negative_rate
        lr = learning_rate if learning_rate is not None else self.learning_rate

        if(negative_rate is not None):        
            neg_ref = np.zeros(negative_rate+1)
            neg_ref[0] = 1
        else:
            neg_ref = self.neg_ref

        ss1 = self.vocab_code_map[s1]
        ss2 = self.vocab_code_map[s2]           

        negs = self.get_negs(nr, [ss2])
        samples = [ss2]+negs
        in_vec = self.W1[ss1]
        out_vecs = self.W2[samples]
        scores = np.dot(in_vec, out_vecs.T)
        fb = expit(scores)
        gb = (neg_ref - fb)*lr
        self.W2[samples] += np.outer(gb, in_vec)
        self.W1[ss1] += gb.dot(out_vecs)

    def get_rank(self, s1, s2, top_k):
        qv = self.W1[s1]
        idv = qv.dot(self.W1[s2])
        bigger = 0
        for v in self.vs:
            bigger += np.sum(v.dot(qv)>idv)
            if(bigger>top_k):
                break
        if bigger >= top_k:
            return None
        else:
            # min value should be one
            return bigger + 1
        
    def get_embed(self):
        return self.W1, self.vocab_code_map

    def write_embed(self, file):
        reverse_map = {v:k for k,v in self.vocab_code_map.items()}
        out = pd.DataFrame(self.W1).reset_index()
        out['index'] = out['index'].map(reverse_map)
        out.to_csv(file, sep=' ', header=False, index=False)
    
    def get_negs(self, num, exclude):
        r = [v for v in np.random.choice(self.vocab_codes, num+len(exclude)+5, p=self.noise_dist_weights) if v not in exclude][:num]
        while(len(r) < num):
            nr = np.random.choice(self.vocab_codes, 1, p=self.noise_dist_weights)[0]
            if(nr not in exclude):
                r.append(nr)
        return r

    def update_noise_dist(self, appearences):
        if(appearences is not None):
            vocab_code_list = [self.vocab_code_map[v] for v in appearences]
            bin_count = np.bincount(vocab_code_list)
            self.noise_dist_counts = np.concatenate([bin_count, np.zeros(len(self.vocab_codes) - np.max(vocab_code_list) - 1, dtype=np.int64)])
        counts = self.noise_dist_counts**self.ns_exponent
        self.noise_dist_weights = counts/np.sum(counts)
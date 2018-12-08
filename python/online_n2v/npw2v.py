import numpy as np
from scipy.special import expit
import pandas as pd
import networkx as nx

class NPWord2Vec:
    def __init__(
        self,
        embedding_dim,
        learning_rate=0.05,
        negative_rate = 5,
        ns_exponent=0.75,
        loss="logsigmoid",
        mirror=False,
        onlymirror=False,
        init='gensim',
        log_updates=False,
        window=10
    ):
        #params
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.negative_rate = negative_rate
        self.ns_exponent = ns_exponent
        self.loss = loss 
        self.mirror = mirror
        self.onlymirror = onlymirror
        self.init = init
        self.log_updates = log_updates
        self.window = window

        #etc
        self.neg_ref = np.zeros(negative_rate+1)
        self.neg_ref[0] = 1
        self.seen = nx.DiGraph()

        if(self.log_updates):
            self.update_sizes = []
            self.updates = []

    def set_vocabulary(self,vocabulary):
        self.vocabulary = np.unique(list(vocabulary))
        self.vocab_size = len(self.vocabulary)
        self.vocab_codes = np.arange(self.vocab_size)
        self.vocab_code_map = dict(zip(self.vocabulary, self.vocab_codes))

        if(self.init == 'gensim'):
            self.W1 = (np.random.rand(self.vocab_size, self.embedding_dim)-0.5)/self.embedding_dim
            self.W2 = np.zeros((self.vocab_size, self.embedding_dim))
        elif(self.init == 'uniform'):
            self.W1 = (np.random.rand(self.vocab_size, self.embedding_dim)-0.5)/100
            self.W2 = (np.random.rand(self.vocab_size, self.embedding_dim)-0.5)/100
        self.noise_dist_counts = np.zeros(self.vocab_size)+0.01
        self.update_noise_dist(self.vocabulary)

    def train_sentence(self, sentence, window=None, learning_rate=None, negative_rate=None):
        window = window if window is not None else self.window
        l = len(sentence)
        if(l==1):
            return
        pairs = []
        for s1 in range(l):
            if(window!=2):
                reduced_window = np.random.randint(1,window-1)
            else:
                reduced_window = window
            for s2 in range(max(0,s1-reduced_window),min(l,s1+reduced_window+1)):
                if(s1!=s2):
                    pairs.append((sentence[s1],sentence[s2]))
        self.train_pairs(pairs, False, False, learning_rate, negative_rate)

    def train_pairs(self, pairs, onlymirror=None, mirror=None, learning_rate = None, negative_rate = None):
        onlymirror = onlymirror if onlymirror is not None else self.onlymirror
        mirror = mirror if mirror is not None else self.mirror
        if(not onlymirror):
            for (s1,s2) in pairs:
               self.train_pair(s1,s2,learning_rate,negative_rate)
        if(mirror):
            for (s2,s1) in pairs:
                self.train_pair(s1,s2,learning_rate,negative_rate)

    def train_pair(self, s1, s2,learning_rate=None,negative_rate=None):
        if(self.log_updates):
            self.updates.append((s1,s2))

        nr = negative_rate if negative_rate is not None else self.negative_rate
        lr = learning_rate if learning_rate is not None else self.learning_rate

        if(negative_rate is not None):
            neg_ref = np.zeros(negative_rate+1)
            neg_ref[0] = 1
        else:
            neg_ref = self.neg_ref

        ss1 , ss2 = self.code_pair(s1,s2)

        negs = self.get_negs(nr, [ss2])
        samples = [ss2]+negs

        if(self.loss == "logsigmoid"):
            self.do_update_logsigmoid(ss1, samples, neg_ref, lr)
        else:
            self.do_update_squared(ss1, samples, neg_ref, lr)

    def do_update_logsigmoid(self, ss1, samples, neg_ref, lr):
        in_vec = self.W1[ss1]
        out_vecs = self.W2[samples]

        scores = np.dot(in_vec, out_vecs.T)
        fb = expit(scores)
        gb = (neg_ref - fb)*lr
        if(self.log_updates):
            self.update_sizes.append(np.mean(np.abs(gb)))
        self.W2[samples] += np.outer(gb, in_vec)
        self.W1[ss1] += gb.dot(out_vecs)

    def do_update_squared(self, ss1, samples, neg_ref, lr):
        in_vec = self.W1[ss1]
        out_vecs = self.W2[samples]
        scores = np.dot(in_vec, out_vecs.T)

        gb = (neg_ref - scores)*lr
        if(self.log_updates):
            self.update_sizes.append(np.mean(np.abs(gb)))
        self.W2[samples] += np.outer(gb, in_vec)
        self.W1[ss1] += gb.dot(out_vecs)
    
    def get_rank(self, s1, s2, top_k, ids):
        ss1, ss2 = self.code_pair(s1,s2)
        p , q = self.W1[ss1], self.W2[ss2]
        r = p.dot(q)
        if(np.isnan(r)):
            raise Exception("NAN encountered")
        rank = 0
        equal = 0
        for node in ids:
            ss3 = self.vocab_code_map[node]
            qq = self.W2[ss3]
            sc = qq.dot(p)
            if sc > r:
                rank += 1
            if sc == r:
                equal += 1
            if(rank>=top_k): return None
        if(equal != 0):
            return_rank = rank+1+np.random.randint(0,equal)
        else:
            return_rank = rank+1
        if(return_rank > top_k):
            return None
        else:
            return return_rank
    
    """
    def get_rank(self, s1, s2, top_k, ids):
        if top_k == None: return None
        ss1, ss2 = self.code_pair(s1,s2)
        if self.seen.has_edge(ss1,ss2) or self.seen.has_edge(ss2,ss1) or (ss1 not in self.seen.nodes()) or (ss2 not in self.seen.nodes()): return None
        p , q = self.W1[ss1], self.W2[ss2]
        r = p.dot(q)
        if(np.isnan(r)):
            raise Exception("NAN encountered")
        rank = 0
        equal = 0
        ids = [i for i in set(self.seen.nodes()) - set(self.seen.neighbors(ss1)) - set(self.seen.predecessors(ss1)) if not i in [ss1, ss2]]
        #id_chunks = np.array_split(list(ids), 12)
        #for ss3s in id_chunks:
        #    if(len(ss3s)==0): continue
        #    qq = self.W2[ss3s]
        #    sc = qq.dot(p)
        #    rank += np.sum(~(sc <= r))
        #    equal += np.sum(sc == r)
        #    if(rank>=top_k): return None
        for ss3 in ids:
            qq = self.W2[ss3]
            sc = qq.dot(p)
            if sc <= r:
                rank += 1
                if sc == r:
                    equal += 1
        if(equal != 0):
            return_rank = rank+1+np.random.randint(0,equal)
        else:
            return_rank = rank+1
        if(return_rank > top_k):
            return None
        else:
            return return_rank
    """
    def add(self,s1,s2):
        ss1, ss2  = self.code_pair(s1,s2)
        self.seen.add_edge(ss1,ss2)

    def code_pair(self,s1,s2):
        ss1 = self.vocab_code_map[s1]
        ss2 = self.vocab_code_map[s2]
        return ss1,ss2

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

import smart_open
smart_open.open = smart_open.smart_open

from gensim.models import Word2Vec
import numpy as np
import random
import pandas as pd
import pickle
import multiprocessing as mp
import itertools


class Node2Vec:

    algorithm_name = 'Node2Vec'

    class WalkingAlgorithm:

        @staticmethod
        def alias_setup(probs):
            """
            Compute utility lists for non-uniform sampling from discrete distributions.
            Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
            for details
            """
            K = len(probs)
            q = np.zeros(K)
            J = np.zeros(K, dtype=np.int)

            smaller = []
            larger = []
            for kk, prob in enumerate(probs):
                q[kk] = K * prob
                if q[kk] < 1.0:
                    smaller.append(kk)
                else:
                    larger.append(kk)

            while len(smaller) > 0 and len(larger) > 0:
                small = smaller.pop()
                large = larger.pop()

                J[small] = large
                q[large] = q[large] + q[small] - 1.0
                if q[large] < 1.0:
                    smaller.append(large)
                else:
                    larger.append(large)

            return J, q

        @staticmethod
        def alias_draw(J, q):
            """
            Draw sample from a non-uniform discrete distribution using alias sampling.
            """
            K = len(J)

            kk = int(np.floor(np.random.rand() * K))
            if np.random.rand() < q[kk]:
                return kk
            else:
                return J[kk]

        def __init__(self, G, p, q):
            self.G = G
            self.is_directed = self.G.is_directed()
            self.p = p
            self.q = q
            self.alias_edges = None
            self.alias_nodes = None

        def node2vec_walk(self, walk_length, start_node):
            """
            Simulate a random walk starting from start node.
            """
            G = self.G
            alias_nodes = self.alias_nodes
            alias_edges = self.alias_edges

            walk = [start_node]

            while len(walk) < walk_length:
                cur = walk[-1]
                cur_nbrs = sorted(G.neighbors(cur))
                if len(cur_nbrs) > 0:
                    if len(walk) == 1:
                        walk.append(cur_nbrs[self.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                    else:
                        prev = walk[-2]
                        next = cur_nbrs[self.alias_draw(alias_edges[(prev, cur)][0],
                                                   alias_edges[(prev, cur)][1])]
                        walk.append(next)
                else:
                    break

            return walk

        def simulate_walks(self, num_walks, walk_length):
            """
            Repeatedly simulate random walks from each node.
            """
            G = self.G
            walks = []
            nodes = list(G.nodes())
            print('Walk iteration:')
            for walk_iter in range(num_walks):
                print(str(walk_iter + 1), '/', str(num_walks))
                random.shuffle(nodes)
                for node in nodes:
                    walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
            return walks

        def get_alias_edge(self, src, dst):
            """
            Get the alias edge setup lists for a given edge.
            """
            G = self.G
            p = self.p
            q = self.q

            unnormalized_probs = []
            for dst_nbr in sorted(G.neighbors(dst)):
                if dst_nbr == src:
                    unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
                elif G.has_edge(dst_nbr, src):
                    unnormalized_probs.append(G[dst][dst_nbr]['weight'])
                else:
                    unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

            return self.alias_setup(normalized_probs)

        def preprocess_transition_probs(self):
            """
            Preprocessing of transition probabilities for guiding the random walks.

            For nodes get L1 Norm of weight of neighbours
            """
            G = self.G
            is_directed = self.is_directed

            alias_nodes = {}
            counter = 1
            for node in G.nodes():
                unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
                norm_const = sum(unnormalized_probs)
                normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
                alias_nodes[node] = self.alias_setup(normalized_probs)
                counter += 1
                if counter % 5000 == 0:
                    print('Alias node created %s/%s' % (counter, len(G.nodes())))

            alias_edges = {}
            counter = 1
            if is_directed:
                for edge in G.edges():
                    alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            else:

                map_list_1 = [(edge[0], edge[1]) for edge in G.edges()]
                map_list_2 = [(edge[1], edge[0]) for edge in G.edges()]

                pool = mp.Pool()
                alias_edges_list1 = pool.starmap(self.get_alias_edge, map_list_1)
                pool.close()
                pool = mp.Pool()
                alias_edges_list2 = pool.starmap(self.get_alias_edge, map_list_2)
                pool.close()

                for edge1, edge2, alias_edge1, alias_edge2 in zip(map_list_1, map_list_2, alias_edges_list1, alias_edges_list2):
                    alias_edges[edge1] = alias_edge1
                    alias_edges[edge2] = alias_edge2
                    counter += 1
                    if counter % 5000 == 0:
                        print('Alias edge created %s/%s' % (counter, len(G.edges())))

            self.alias_nodes = alias_nodes
            self.alias_edges = alias_edges

    def __init__(self,
                 p=1,
                 q=1,
                 number_walks=10,
                 walk_length=80,
                 dimensions=128,
                 window_embedding=10,
                 n_jobs=None,
                 epoch_embeddings=1,
                 protein_identifier=None,
                 is_directed=False,
                 walk_path=None,
                 vector_path=None):
        self.p = p
        self.q = q
        self.is_directed = is_directed
        self.walking = None
        self._walks = None
        self.embedding_results = None
        self.results = None
        self.df_sim = None
        self.number_walks = number_walks
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.window_embedding = window_embedding
        if n_jobs is None:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = n_jobs
        self.epoch_embeddings = epoch_embeddings
        self.protein_identifier = protein_identifier
        self.walk_path = walk_path
        self.vector_path = vector_path
        self.modified_G = None

    def get_params(self):
        params = {
            'p': self.p,
            'q': self.q,
            'number_walks': self.number_walks,
            'walk_length': self.walk_length,
            'dimensions': self.dimensions,
            'window_embedding': self.window_embedding,
            'n_jobs': self.n_jobs,
            'epoch_embeddings': self.epoch_embeddings,
            'protein_identifier': self.protein_identifier
        }
        return params

    def set_params(self, p, q, number_walks, walk_length, dimensions,
                   window_embedding, n_jobs, epoch_embeddings, protein_identifier):
        self.p = p
        self.q = q
        self.number_walks = number_walks
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.window_embedding = window_embedding
        self.n_jobs = n_jobs
        self.epoch_embeddings = epoch_embeddings
        self.protein_identifier = protein_identifier

    def copy(self):
        return Node2Vec(**self.get_params())

    def preprocess_transition_probs(self, G):
        self.walking = self.WalkingAlgorithm(G, self.p, self.q)
        print('Start preprocess_transition_probs')
        self.walking.preprocess_transition_probs()
        print('Finish preprocess_transition_probs')

    def get_walks(self, index):
        print('Starting walker %s' % index)
        walks = self.walking.simulate_walks(num_walks=int(np.ceil(self.number_walks/self.n_jobs)), walk_length=self.walk_length)
        print('Finished walker %s' % index)
        return walks

    def simulate_walks(self, G):
        self.walking = self.WalkingAlgorithm(G, self.p, self.q)
        print('Start preprocess_transition_probs')
        self.walking.preprocess_transition_probs()
        if self.n_jobs == 1:
            self._walks = self.walking.simulate_walks(num_walks=self.number_walks, walk_length=self.walk_length)
        else:
            pool = mp.Pool(self.n_jobs)
            list_of_walks = pool.map(self.get_walks, range(self.n_jobs))
            pool.close()
            self._walks = list(itertools.chain.from_iterable(list_of_walks))
            self._walks = [item[:self.walk_length] for item in self._walks]

    def learn_embeddings(self):
        """
        Learn embeddings by optimizing the Skipgram objective using SGD.
        """
        walks_map = [list(map(str, walk)) for walk in self._walks]
        self.embedding_results = Word2Vec(
            walks_map,
            size=self.dimensions,
            window=self.window_embedding,
            min_count=0, sg=1,
            workers=self.n_jobs,
            iter=self.epoch_embeddings)

    def run(self, G, seed_nodes=None):

        if self.vector_path is not None:

            with open(self.vector_path, 'rb') as f:
                self.embedding_results = pickle.load(f)

        if self.embedding_results is None:

            if self._walks is None:

                if self.walk_path is not None:
                    with open(self.walk_path, 'rb') as f:
                        self._walks = pickle.load(f)
                else:
                    self.simulate_walks(G)

            self.learn_embeddings()

        if seed_nodes is not None:
            list_of_sim = list()
            for node in G.nodes:
                node_sim = list()
                node_vec = self.embedding_results.wv.get_vector(str(node))
                for seed_node in seed_nodes:
                    seed_vec = self.embedding_results.wv.get_vector(str(seed_node))
                    node_sim.append(np.inner(node_vec, seed_vec) / (np.linalg.norm(node_vec) * np.linalg.norm(seed_vec)))
                list_of_sim.append(node_sim)
            df_sim = pd.DataFrame.from_records(list_of_sim, index=G.nodes, columns=seed_nodes)
            df_sim['max_sim'] = [np.max(row[1:]) for row in df_sim.itertuples()]

            self.df_sim = df_sim
            if self.protein_identifier is None:
                self.results = {key: value
                                for key, value in zip(G.nodes, self.df_sim.max_sim)}
            else:
                self.results = {key: value
                                for key, value in zip(G.nodes, self.df_sim.max_sim) if self.protein_identifier in key}

    def get_results_df(self, sorting=True, column_name='max_sim'):
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        results_df.columns = [column_name]
        if sorting:
            results_df.sort_values(by=column_name, inplace=True, ascending=False)
        return results_df

    def get_metrics(self, metrics_function, key=None):
        return metrics_function(self.results, key)


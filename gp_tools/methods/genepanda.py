from gp_tools.IO import pickle_dump_to_file
import pandas as pd
import networkx as nx
import multiprocessing as mp
import pickle
import os
import numpy as np


class GenePanda:

    algorithm_name = 'GenePanda'

    counter = mp.Value('i', 0)

    @staticmethod
    def normalize(mat):

        row_mean = np.mean(mat, axis=1)
        if isinstance(row_mean, np.matrix):
            row_mean = row_mean.A1

        for i in range(len(row_mean)):
            if row_mean[i] == 0:
                row_mean[i] = 1

        normalizer = np.power(row_mean, -0.5)
        mat_norm = mat.copy()

        for i in range(len(normalizer)):
            mat_norm[i, :] = mat_norm[i, :] * normalizer[i]
        for i in range(len(normalizer)):
            mat_norm[:, i] = mat_norm[:, i] * normalizer[i]
        return mat_norm

    def __init__(self,
                 spl_dump=None):
        self.spl_dump = spl_dump
        self.largest_cc = None
        self.largest_cc_nodes = None
        self.spl_matrix_normalized = None
        self.mean_norm_distance_global = None
        self.mean_norm_distance_seed = None
        self.largest_cc_nodes_index = None
        self.seed_index = None
        self.distance_global_minus_seed = None
        self.results = None

    def copy(self):
        return GenePanda(**self.get_params())

    def set_params(self, spl_dump):
        self.spl_dump = spl_dump

    def get_params(self):
        params = {
            'spl_dump': self.spl_dump
        }
        return params

    def _spl(self, node):
        length_dict = nx.single_source_shortest_path_length(self.largest_cc, node)
        with self.counter.get_lock():
            self.counter.value += 1
            print('Ndoes finished (%s/%s)' % (self.counter.value, self.largest_cc.number_of_nodes()))

        return np.fromiter([length_dict[target_node] for target_node in self.largest_cc.nodes],
                           dtype='float')

    def setup_spl_matrix(self, G):

        self.largest_cc = G.subgraph(max(nx.connected_components(G), key=len))
        self.largest_cc_nodes = list(self.largest_cc.nodes)
        self.largest_cc_nodes_index = {node: index for index, node in enumerate(self.largest_cc_nodes)}

        if os.path.exists(self.spl_dump):

            with open(self.spl_dump, 'rb') as f:
                self.spl_matrix_normalized = pickle.load(f)

        else:

            pool = mp.Pool(mp.cpu_count())
            shortest_path_lengths = pool.map(self._spl, self.largest_cc_nodes)
            pool.close()

            self.spl_matrix_normalized = self.normalize(np.array(shortest_path_lengths))

            pickle_dump_to_file(self.spl_matrix_normalized, self.spl_dump)

        if np.trace(self.spl_matrix_normalized) != 0:
            raise ValueError('Order of the spl matrix is probably wrong.')
        self.mean_norm_distance_global = np.mean(self.spl_matrix_normalized, axis=1)

    def run(self, G, seed_nodes):

        if os.path.exists(self.spl_dump):
            self.largest_cc = G.subgraph(max(nx.connected_components(G), key=len))
            self.largest_cc_nodes = list(self.largest_cc.nodes)
            self.largest_cc_nodes_index = {node: index for index, node in enumerate(self.largest_cc_nodes)}

            with open(self.spl_dump, 'rb') as f:
                self.spl_matrix_normalized = pickle.load(f)

            if np.trace(self.spl_matrix_normalized) != 0:
                raise ValueError('Order of the spl matrix is probably wrong.')
            self.mean_norm_distance_global = np.mean(self.spl_matrix_normalized, axis=1)

        if self.spl_matrix_normalized is None:
            raise EnvironmentError('Shortest path length matrix not setup: '
                                   'Run self.setup_spl_matrix() first')

        if set(max(nx.connected_components(G), key=len)) != set(self.largest_cc.nodes):
            raise IOError('Different graph used for '
                          'Shortest path length matrix')

        self.seed_index = [self.largest_cc_nodes_index[seed] for seed in seed_nodes]

        self.mean_norm_distance_seed = np.mean(self.spl_matrix_normalized[:, self.seed_index], axis=1)
        self.distance_global_minus_seed = self.mean_norm_distance_global - self.mean_norm_distance_seed

        self.results = {node: distance for node, distance in zip(self.largest_cc_nodes, self.distance_global_minus_seed)}

    def get_results_df(self, sorting=True, column_name='GenePanda_value'):
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        results_df.columns = [column_name]
        if sorting:
            results_df.sort_values(by=column_name, inplace=True, ascending=False)
        return results_df

    def get_metrics(self, metrics_function, key=None):
        return metrics_function(self.results, key)


#
# G = nx.fast_gnp_random_graph(1000, 0.05)
# gp = GenePanda(spl_dump='test')
# gp.setup_spl_matrix(G)
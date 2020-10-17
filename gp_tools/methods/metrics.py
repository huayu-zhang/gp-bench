import networkx as nx
import numpy as np
from gp_tools.methods.randomwalk import RandomWalk
from gp_tools.nx_extension.parallel_centralities import betweenness_centrality_parallel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp


# Metrics for evaluation

class ModularityAnalysis:

    def __init__(self, G):
        self.G = G
        self.communities = None
        self.modularity_scores = None
        self.community_dict = None
        self.community_dict_ordered = None
        self.adjacency_matrix = None
        self.n_stud = None
        self.node_degree = None
        self.degree_outer_matrix = None
        self.modularity_matrix_B = None
        self.membership_matrix_S = None
        self.modularity_scores = None
        self.graph_modularity = None

    def compute_communities(self):
        self.communities = nx.algorithms.community.greedy_modularity_communities(self.G)

    def matmul_vtAv(self, vector, matrix):
        return vector @ matrix @ vector

    def compute_modularity(self):

        self.community_dict = {node: i for i, nodes in enumerate(self.communities) for node in list(nodes)}
        self.community_dict_ordered = {node: self.community_dict.get(node) for node in self.G.nodes}

        self.adjacency_matrix = nx.to_numpy_matrix(self.G)
        self.n_stud = nx.number_of_edges(self.G) * 2

        self.node_degree = np.array([degree for node, degree in self.G.degree])
        self.degree_outer_matrix = np.outer(self.node_degree, self.node_degree) - np.diag(self.node_degree ** 2)

        self.modularity_matrix_B = self.adjacency_matrix - self.degree_outer_matrix / self.n_stud

        self.membership_matrix_S = np.array(
            [(np.array(list(self.community_dict_ordered.values())) == community).astype(float) for community in
             range(self.communities.__len__())]
        )

        self.modularity_scores = [self.matmul_vtAv(s, self.modularity_matrix_B) for s in self.membership_matrix_S]

        self.graph_modularity = sum(self.modularity_scores) / self.n_stud

        return self.graph_modularity


class PairwiseMetrics:

    def shortest_path_length(self, u, v):
        return nx.shortest_path_length(self.G, u, v)

    def common_neighbor_number(self, u, v):
        number = 2 * len([node for node in nx.common_neighbors(self.G, u, v)])/(len(self.G[u]) + len(self.G[v]))
        return number

    @staticmethod
    def rwr_distance_pairwise(G, seed_nodes):

        rwr_distance = list()

        for u in seed_nodes:
            rw = RandomWalk().copy()
            rw.run(G, u)
            node_results = rw.results.get(u)
            rwr_distance.append([np.negative(np.log10(rw.results.get(v) / node_results)) for v in seed_nodes])

        rwr_distance_mat = (np.array(rwr_distance) + np.array(rwr_distance).transpose()) / 2

        rwr_distance_df = pd.DataFrame(rwr_distance_mat, index=seed_nodes, columns=seed_nodes)

        for i in range(len(seed_nodes)):
            rwr_distance_df.iloc[i, i] = 0

        return rwr_distance_df

    @staticmethod
    def number_of_paths_below_cutoff(G, seed_nodes, cutoff=4, log10_scale=True):

        adj_mat = nx.adjacency_matrix(G)
        path_number_mat = adj_mat
        node_list = list(G)

        for i in range(cutoff - 1):
            path_number_mat = path_number_mat @ adj_mat

        seed_index = [index for index in range(len(G)) if node_list[index] in seed_nodes]

        seed_path_number_mat = path_number_mat[seed_index, :][:, seed_index].toarray()

        if log10_scale:
            seed_path_number_mat = np.log10(seed_path_number_mat)

        seed_path_number_df = pd.DataFrame(seed_path_number_mat, index=seed_nodes, columns=seed_nodes)

        return seed_path_number_df

    def number_of_simple_paths_below_cutoff(self, source, target, cutoff=4, log10_scale=True):

        G = self.G

        number_of_simple_paths = len(list(nx.all_simple_paths(G, source, target, cutoff)))
        if log10_scale:
            if number_of_simple_paths > 0:
                number_of_simple_paths = 5 - np.log10(number_of_simple_paths)
        return number_of_simple_paths

    def weighted_number_of_simple_paths_below_cutoff(self, source, target, cutoff=4, weight=10):

        G = self.G

        simple_paths_lengths = np.array([len(path) for path in nx.all_simple_paths(G, source, target, cutoff)])
        weighted_number_of_simple_paths = sum(np.power(weight, cutoff - simple_paths_lengths + 1))

        if weighted_number_of_simple_paths > 0:
            weighted_number_of_simple_paths = 1/np.log10(weighted_number_of_simple_paths)
        return weighted_number_of_simple_paths

    def __init__(self, G, seed_nodes):
        self.G = G
        self.seed_nodes = seed_nodes
        self.metrics_dict = None

        self.available_metrics = {
            'shortest_path_length': self.shortest_path_length,
            'common_neighbor_number': self.common_neighbor_number,
            'number_of_simple_paths_below_cutoff': self.number_of_simple_paths_below_cutoff,
            'weighted_number_of_simple_paths_below_cutoff': self.weighted_number_of_simple_paths_below_cutoff
        }

        self.available_wrapped_metrics = {
            'RWR_distance': self.rwr_distance_pairwise,
            'number_of_paths_below_cutoff': self.number_of_paths_below_cutoff
        }

    def compute_metrics(self, metrics='all', wrapped_metrics='all'):
        self.metrics_dict = dict()

        # Metrics that need pairwise execution
        pairwise_map_list = [(node1, node2) for node1 in self.seed_nodes for node2 in self.seed_nodes]
        n_seeds = len(self.seed_nodes)
        n_results = len(pairwise_map_list)

        if metrics == 'all':
            for metric_name, function in self.available_metrics.items():
                pool = mp.Pool(mp.cpu_count())
                results = pool.starmap(function, pairwise_map_list)
                pool.close()

                results_in_chunks = [results[i:i+n_seeds] for i in range(0, n_results, n_seeds)]

                self.metrics_dict[metric_name] = pd.DataFrame(results_in_chunks,
                                                              index=self.seed_nodes, columns=self.seed_nodes)
        elif metrics is None:
            pass
        else:
            for metric in metrics:
                pool = mp.Pool(mp.cpu_count())
                results = pool.starmap(self.available_metrics[metric], pairwise_map_list)
                pool.close()

                results_in_chunks = [results[i:i+n_seeds] for i in range(0, n_results, n_seeds)]

                self.metrics_dict[metric] = pd.DataFrame(results_in_chunks,
                                                         index=self.seed_nodes, columns=self.seed_nodes)

        # Metrics that have wrapped pairwise execution inside the function
        if wrapped_metrics == 'all':
            for metric_name, function in self.available_wrapped_metrics.items():
                self.metrics_dict[metric_name] = function(self.G, self.seed_nodes)
        elif wrapped_metrics is None:
            pass
        else:
            for metric in wrapped_metrics:
                self.metrics_dict[metric] = self.available_wrapped_metrics[metric](self.G, self.seed_nodes)

    def get_heatmaps(self, path_or_buf=None, metrics=None, alt_seed_names=None):

        im_list = list()

        if metrics is None:
            metrics = self.metrics_dict.keys()

        for metric_name in metrics:

            if alt_seed_names is not None:
                self.metrics_dict[metric_name].columns = alt_seed_names
                self.metrics_dict[metric_name].index = alt_seed_names

            plt.close()
            im = sns.clustermap(data=self.metrics_dict[metric_name], annot=True)
            im.fig.axes[2].tick_params(axis='x', labelrotation=90)
            im.fig.axes[2].tick_params(axis='y', labelrotation=0)
            im.fig.set_figheight(6)
            im.fig.set_figwidth(6)

            im_list.append(im)

            if path_or_buf is not None:
                im.savefig('_'.join([path_or_buf, metric_name, '.png']))
            plt.show()
            plt.close()

        return im_list


class NodeMetrics:

    @staticmethod
    def seed_one_hot(G, seed_nodes=None):
        OH_dict = dict(zip(G.nodes, [0] * len(G.nodes)))
        if seed_nodes is not None:
            for seed_node in seed_nodes:
                OH_dict[seed_node] = 1
        return OH_dict

    fast_metrics = {
        'degree_centrality': nx.degree_centrality,
        'eigenvector_centrality': nx.eigenvector_centrality,
        'page_rank': nx.pagerank
    }

    slow_metrics = {
        'betweenness_centrality': betweenness_centrality_parallel
    }

    def __init__(self, G, seed_nodes=None):
        self.G = G
        self.seed_nodes = seed_nodes
        self.metrics_df = pd.DataFrame.from_dict(self.seed_one_hot(self.G, self.seed_nodes),
                                                 orient='index', columns=['is_seed'])
        self.metrics_df_norm = None
        self.metrics_melt = None
        self.metrics_summary = None

    def compute_fast_metrics(self):

        for metric_name, function in self.fast_metrics.items():
            self.metrics_df[metric_name] = function(self.G).values()

    def compute_slow_metrics(self):

        for metric_name, function in self.slow_metrics.items():
            self.metrics_df[metric_name] = function(self.G).values()

    def add_manual_metrics(self, metric_values, metric_name):
        self.metrics_df[metric_name] = metric_values

    def get_metrics_df(self):
        return self.metrics_df

    def get_metrics_summary(self):

        if self.seed_nodes is not None:
            self.metrics_summary = pd.concat([
                self.metrics_df.describe().iloc[[1, 2, 5], ],
                self.metrics_df.loc[self.metrics_df.is_seed == 1, ].describe().iloc[[1, 2, 5], ]
            ])
            self.metrics_summary.index = ['graph_mean', 'graph_std', 'graph_median',
                                          'seed_mean', 'seed_std', 'seed_median']
        else:
            self.metrics_summary = self.metrics_df.describe().iloc[[1, 2, 5], ]

        return self.metrics_summary

    def visualize_metrics(self, path_or_buf=None, normalize=True, **kwargs):

        self.metrics_df_norm = self.metrics_df.copy()
        for column in self.metrics_df_norm.columns:
            if column != 'is_seed':
                self.metrics_df_norm[column] = self.metrics_df_norm[column]/np.mean(self.metrics_df_norm[column])

        if normalize:
            self.metrics_melt = self.metrics_df_norm.melt(id_vars=['is_seed'])
        else:
            self.metrics_melt = self.metrics_df.melt(id_vars=['is_seed'])

        im = sns.boxplot(x='variable', y='value', color='grey', data=self.metrics_melt)
        if self.seed_nodes is not None:
            sns.stripplot(x='variable', y='value', color='red',
                          data=self.metrics_melt.loc[self.metrics_melt.is_seed == 1, ])

        im.set(**kwargs)
        plt.show()

        if path_or_buf is not None:
            im.get_figure().savefig(path_or_buf)

        plt.close()
        return im


class GraphMetrics:

    @staticmethod
    def largest_component_size(G):
        connected_components_size = [len(c) for c in nx.connected_components(G)]
        return max(connected_components_size)

    @staticmethod
    def graph_edge_types(G):
        edge_types = set(nx.get_edge_attributes(G, 'type').values())
        return '|'.join(list(edge_types))

    def __init__(self, list_G, names_G=None):
        self.list_G = list_G
        self.names_G = names_G
        self.available_metrics = {
            'number_of_nodes': nx.number_of_nodes,
            'number_of_edges': nx.number_of_edges,
            'number_connected_components': nx.number_connected_components,
            'largest_connected_component_size': self.largest_component_size
        }
        self.metrics_df = None

    def compute_metrics(self, metrics=None):
        if metrics is None:
            metrics = self.available_metrics.keys()

        self.metrics_df = pd.DataFrame([self.graph_edge_types(graph) for graph in self.list_G],
                                       columns=['graph_type'])

        if (self.names_G is not None) and (len(self.names_G) == len(self.list_G)):
            self.metrics_df.index = self.names_G

        for metric in metrics:
            self.metrics_df[metric] = [self.available_metrics[metric](graph) for graph in self.list_G]

    def get_metrics_df(self):
        return self.metrics_df


class ModelMetrics:

    @staticmethod
    def rank(results, keys=None):
        """
        :param results: In the form of {node1: value1, ...}
        :param keys: ['node1', ...]
        :return: {node1: rank1, ...]
        """
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.columns = ['value']
        results_df['ranking'] = results_df.value.rank(ascending=False)
        if keys is None:
            rank_dict = dict(zip(results_df.index, results_df.ranking))
        else:
            rank_dict = dict()
            for key in keys:
                rank_dict[key] = results_df.ranking[key]

        return rank_dict

    @staticmethod
    def mean_rank(results):
        """
        :param results: [tune_iter1 -> {cv1: value1, ...}, {}, ...]
        :return: [mean_rank1, mean_rank2,...]
        """
        return [np.mean([*tune_iter.values()]) for tune_iter in results]

    @staticmethod
    def minus_rank_ratio(results):
        return [np.mean(
            -50/np.array([*tune_iter.values()])
        ) for tune_iter in results]

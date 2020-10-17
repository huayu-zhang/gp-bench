import warnings
import networkx as nx
import pandas as pd
import itertools
import time


def param_to_name(param):
    name = '_'.join(['%s-%s' % (key, param[key]) for key in param])
    name = name.replace('.', '').replace('/', '')
    return name

def name_to_param(name):
    return dict([tuple(pair.split('-')) for pair in name.split('_')])


def preprocess_seed_target(G, seed_nodes, target_nodes):

    """
    :param G: nx graph
    :param seed_nodes: seed_nodes
    :param target_nodes: target nodes where seed-target relationship is the objective of prediction
    :return: the processed graph
    """

    if isinstance(seed_nodes, str):
        seed_nodes = [seed_nodes]

    if isinstance(target_nodes, str):
        target_nodes = [target_nodes]

    # Check if target nodes in Graph
    # If so:
    # converting target nodes to '$$$target_node'
    # supplement seed-target edges
    mapping = {node: '$$$target_node' for node in target_nodes if node in G.nodes}

    if len(mapping) > 0:
        G = nx.relabel_nodes(G, mapping=mapping)

        # Supplement seed-target edges
        edge_supplement = [(*edge, {'type': 'seed-target', 'weight': 1})
                           for edge in itertools.product(seed_nodes, ['$$$target_node'])]
        G.add_edges_from(edge_supplement)

    return G


def init_value_by_seed(G, seed_nodes, user_values=None):
    """
    Input a nx graph and a list of seed nodes, return the dict of nodes and init values.
    Init value = 0 if not seed, = 1/seed number if seed, if not defined by user_values
    The dist can be used directly in various RW-based functions as personalization values
    :param G: nx graph
    :param seed_nodes: list of seed nodes
    :param user_values: list of seed values if specific values are to be assigned
    :return: dict { gene1: 0, gene2: 1/N, ...}
    """
    if not isinstance(seed_nodes, list):
        seed_nodes = {seed_nodes}
    seed_nodes = set(seed_nodes)
    seed_not_in_graph = seed_nodes.difference(set(G.nodes))
    n_seed_nodes = seed_nodes.__len__() - seed_not_in_graph.__len__()
    init_value_dict = dict.fromkeys(G.nodes, float(0))

    if user_values is None:

        if n_seed_nodes:
            seed_value = 1/n_seed_nodes
        else:
            warnings.warn('None of seed nodes in graph.')
            seed_value = 1/G.number_of_nodes()

        for node in seed_nodes:
            if node not in seed_not_in_graph:
                init_value_dict[node] = seed_value

    else:
        for node, value in zip(seed_nodes, user_values):
            if node not in seed_not_in_graph:
                init_value_dict[node] = value

    if seed_not_in_graph.__len__():
        warnings.warn("Not all seed nodes in the graph: %s" % seed_not_in_graph)

    return init_value_dict


# Computing time simulation

class TimeSimulationNetwork:

    @staticmethod
    def random_graph_set(n_nodes, p_edges):
        """
        The function to generate a set of random graphs

        :param p_edges: list of number of nodes
        :param n_nodes: list of probability of edges
        :return: list of graphs using combinations of n and p
        """

        random_graphs = []
        for n in n_nodes:
            for p in p_edges:
                random_graphs.append(nx.erdos_renyi_graph(n=n, p=p))
        return random_graphs

    @staticmethod
    def expand_grid_local(n_nodes, p_edges):

        if not isinstance(n_nodes, list):
            n_nodes = list(n_nodes)
        if not isinstance(p_edges, list):
            p_edges = list(p_edges)

        iter_df = pd.DataFrame.from_records(itertools.product(n_nodes, p_edges), columns=['n_nodes', 'p_edges'])
        return iter_df

    def __init__(self,
                 function,
                 n_nodes=None,
                 p_edges=None,
                 max_iter_time=60
                 ):
        if n_nodes is None:
            self.n_nodes = [int(i) for i in [1e2, 3e2, 1e3]]
        else:
            self.n_nodes = n_nodes

        if p_edges is None:
            self.p_edges = [0.0005, 0.001]
        else:
            self.p_edges = p_edges

        self.function = function
        self.simulation_grid = self.expand_grid_local(self.n_nodes, self.p_edges)
        self._simulation_graphs = [nx.subgraph(graph, nbunch=max(nx.connected_components(graph), key=len))
                                   for graph in self.random_graph_set(self.n_nodes, self.p_edges)]
        self.simulation_grid['n_edges'] = [graph.number_of_edges() for graph in self._simulation_graphs]
        self.computing_time = list()
        self.computing_time_df = self.simulation_grid.copy()
        self.max_iter_time = max_iter_time

    def time_it(self, **kwargs):
        time_current_iter = 0

        for graph in self._simulation_graphs:
            if time_current_iter <= self.max_iter_time:
                start_time = time.time()
                self.function(graph, **kwargs)
                time_current_iter = time.time() - start_time
                self.computing_time.append(time_current_iter)
            else:
                self.computing_time.append(None)

        self.computing_time_df['computing_time'] = self.computing_time
        self.computing_time_df['per_1000_node'] = \
            self.computing_time_df.computing_time / self.computing_time_df.n_nodes * 1000
        self.computing_time_df['per_1000_edge'] = \
            self.computing_time_df.computing_time / self.computing_time_df.n_edges * 1000
        self.computing_time_df['per_1000_node_edge_sum'] = self.computing_time_df.computing_time / (
                self.computing_time_df.n_edges + self.computing_time_df.n_nodes) * 1000
        self.computing_time_df['per_1000_node_edge_product'] = self.computing_time_df.computing_time / (
                self.computing_time_df.n_edges * self.computing_time_df.n_nodes) * 1000

    def get_computing_time(self):
        return self.computing_time_df




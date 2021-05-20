from sklearn.preprocessing import normalize
from gp_tools.methods.utils import init_value_by_seed
import networkx as nx
import pandas as pd
import numpy as np


class RandomWalk:

    algorithm_name = 'Random_Walk'

    def __init__(self,
                 gamma=0.5,
                 tol=1e-8,
                 max_iter=100,
                 protein_identifier=None
                 ):
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.norm_function = normalize
        self.protein_identifier = protein_identifier
        self.node_init_values = dict()
        self.node_final_values = dict()
        self.results = None

    def copy(self):
        return RandomWalk(**self.get_params())

    def set_params(self, gamma, tol, max_iter, protein_identifier):
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.protein_identifier = protein_identifier

    def get_params(self):
        params = {
            'gamma': self.gamma,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'protein_identifier': self.protein_identifier
            }
        return params

    def run(self, G,
            seed_nodes=None):
        """
        :param G: Input graph
        :param seed_nodes: Seeds for RWR algorithm
        """
        if seed_nodes is not None:

            if isinstance(seed_nodes, str):
                seed_nodes = [seed_nodes]

            self.node_init_values = init_value_by_seed(G, seed_nodes)

        else:
            self.node_init_values = init_value_by_seed(G, G.nodes)

        self.node_final_values = self.node_init_values.copy()

        p0 = np.array([*self.node_init_values.values()])
        W_norm = normalize(nx.adjacency_matrix(G), norm='l1', axis=0).toarray()
        residue = 1
        n_iter = 0
        pt = p0.copy()

        while (residue > self.tol) & (n_iter <= self.max_iter):
            pt_minus = pt.copy()
            pt = (1 - self.gamma) * np.matmul(W_norm, pt_minus) + self.gamma * p0
            residue = np.sqrt(np.sum((pt - pt_minus)**2))
            n_iter += 1

        if n_iter >= self.max_iter:
            Warning("Algorithm has not converged!")

        for key, value in zip(self.node_init_values.keys(), pt):
            self.node_final_values[key] = value

        if self.protein_identifier is None:
            self.results = self.node_final_values
        else:
            self.results = {key: self.node_final_values[key]
                            for key in self.node_final_values if self.protein_identifier in key}

    def get_results_df(self, sorting=True, column_name='rw_value'):
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        results_df.columns = [column_name]
        if sorting:
            results_df.sort_values(by=column_name, inplace=True, ascending=False)
        return results_df

    def get_metrics(self, metrics_function, key=None):
        return metrics_function(self.results, key)

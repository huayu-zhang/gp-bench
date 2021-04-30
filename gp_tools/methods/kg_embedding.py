import networkx as nx
import pandas as pd
import os
from scipy import spatial


class TransE:

    algorithm_name = 'KGEmbedding'

    @staticmethod
    def cosine_sim(v1, v2):
        return 1 - spatial.distance.cosine(v1, v2)

    def __init__(self,
                 params={},
                 protein_identifier='ENS'
                 ):

        self.params = params
        self.protein_identifier = protein_identifier
        self.embeddings = {}
        self.results = {}

    def copy(self):
        return TransE(params=self.params,
                      protein_identifier=self.protein_identifier)

    def get_params(self):
        return self.params

    def set_params(self, params={}):
        self.params = params

    def run(self, G,
            seed_nodes=None):
        """
        :param G: Input graph in Networkx Format
        :param seed_nodes: Seeds for RWR algorithm
        """
        # Here convert networkx to the format used for dgl-ke

        # Here invoke the command line used for running the algorithm
        os.system('Command line used for the method with params set as string formatting %s' % self.params)

        # Some process here to convert algorithm output to dict of embeddings {node1: v1, node2: v2...}
        self.embeddings = {}

        # Calculate similarity score
        if seed_nodes is not None:
            for node in self.embeddings:
                self.results[node] = max([
                    self.cosine_sim(self.embeddings[node], self.embeddings[seed_node]) for seed_node in seed_nodes])

    def get_results_df(self, sorting=True, column_name='max_sim'):
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        results_df.columns = [column_name]
        if sorting:
            results_df.sort_values(by=column_name, inplace=True, ascending=False)
        return results_df

    def get_metrics(self, metrics_function, key=None):
        return metrics_function(self.results, key)

import numpy as np
import networkx as nx
import pandas as pd
import time


class IDLP:

    algorithm_name = 'IDLP'

    @staticmethod
    def normalize_1(mat):
        """
        Alternative function for D^-0.5 W D^-0.5 normalization.
        Slower but stable
        :param mat: 2d array
        :return: normalized 2d array
        """
        rowsum = np.sum(mat, axis=1)
        for i in range(len(rowsum)):
            if rowsum[i] == 0:
                rowsum[i] = 1
        diag_normalizer = np.diagflat(np.power(rowsum, -0.5))
        return diag_normalizer @ mat @ diag_normalizer

    @staticmethod
    def normalize(mat):
        """
        Function for D^-0.5 W D^-0.5 normalization.
        Fast implementation but not entirely stable
        :param mat: 2d array
        :return: normalized 2d array
        """

        rowsum = np.sum(mat, axis=1)
        if isinstance(rowsum, np.matrix):
            rowsum = rowsum.A1

        for i in range(len(rowsum)):
            if rowsum[i] == 0:
                rowsum[i] = 1

        normalizer = np.power(rowsum, -0.5)
        mat_norm = mat.copy()

        for i in range(len(normalizer)):
            mat_norm[i, :] = mat_norm[i, :] * normalizer[i]
        for i in range(len(normalizer)):
            mat_norm[:, i] = mat_norm[:, i] * normalizer[i]
        return mat_norm

    def __init__(self,
                 alpha=0.5,
                 gamma=1,
                 tol=1e-8,
                 max_iter=10,
                 protein_identifier='ENS',
                 target_disease='$$$target_node'):
        self.alpha = alpha
        self.beta = 1 - self.alpha
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.protein_identifier = protein_identifier
        self.target_disease = target_disease
        self.target_disease_specified = self.target_disease is not None
        self.final_Y = None
        self.results = None
        self._seed = 4000

    def copy(self):
        return IDLP(**self.get_params())

    def set_params(self, alpha, gamma, tol, max_iter, protein_identifier, target_disease):
        self.alpha = alpha
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.protein_identifier = protein_identifier
        self.target_disease = target_disease

    def get_params(self):
        params = {
            'alpha': self.alpha,
            'gamma': self.gamma,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'protein_identifier': self.protein_identifier,
            'target_disease': self.target_disease
        }
        return params

    def run(self, G,
            seed_nodes=None):

        if seed_nodes is not None:
            if self.target_disease is not '$$$target_node':
                for node in seed_nodes:
                    G.add_edge(node, self.target_disease)

        # Reorder the node_list in G as [protein 1,,,protein n, disease 1,,,disease n]
        protein_nodes = [node for node in G.nodes if self.protein_identifier in node]
        disease_nodes = [node for node in G.nodes if self.protein_identifier not in node]

        n_protein = len(protein_nodes)
        n_disease = len(disease_nodes)

        nodes_reordered = protein_nodes + disease_nodes
        adjacency_reordered = nx.adjacency_matrix(G, nodelist=nodes_reordered).toarray()

        # Extract ppi matrix and disease matrix
        S1_protein_adj = adjacency_reordered[:n_protein, :n_protein]
        S2_disease_adj = adjacency_reordered[n_protein:, n_protein:]
        Y0_gda = adjacency_reordered[:n_protein, n_protein:]
        np.random.seed(self._seed)
        Y_current = np.random.rand(n_protein, n_disease)

        print('Adjacency matrices extracted.')

        # Get normailized PPI and Dsim network
        S1_protein_adj = self.normalize(S1_protein_adj)
        S2_disease_adj = self.normalize(S2_disease_adj)

        print('Adjacency matrices normalized.')

        # Get identity matrix of size of S1 and S2
        Diag_1 = np.identity(n_protein)
        Diag_2 = np.identity(n_disease)

        residue = 1
        n_iter = 1

        starting_time = time.time()

        while (residue > self.tol) & (n_iter <= self.max_iter):

            print('Start iter %s' % n_iter)

            Y_previous = Y_current.copy()
            S1_protein_adj = S1_protein_adj + self.gamma * Y_current @ Y_current.transpose()
            print('Finished iter %s.1.1, S1 updated' % n_iter)

            Y_current = self.beta * np.linalg.inv(Diag_1 - self.alpha * S1_protein_adj) @ Y0_gda
            print('Finished iter %s.1.2, Y updated' % n_iter)

            S2_disease_adj = S2_disease_adj + self.gamma * Y_current.transpose() @ Y_current
            print('Finished iter %s.2.1, S2 updated' % n_iter)

            Y_current = self.beta * Y0_gda @ np.linalg.inv(Diag_2 - self.alpha * S2_disease_adj)
            print('Finished iter %s.2.2, Y updated' % n_iter)

            residue = np.sqrt(np.sum((Y_current - Y_previous)**2))
            print('Finished iter %s.2. Residue this iter %s. Time lapsed %s'
                  % (n_iter, residue, time.time() - starting_time))

            n_iter += 1

        self.final_Y = pd.DataFrame(Y_current, columns=disease_nodes, index=protein_nodes)

        if self.target_disease_specified:
            self.results = {index: value for index, value in zip(self.final_Y.index, self.final_Y[self.target_disease])}

    def get_results_df(self, sorting=True, column_name='IDLP'):

        if self.target_disease_specified:
            results_df = pd.DataFrame.from_dict(self.results, orient='index')
            results_df.columns = [column_name]
            if sorting:
                results_df.sort_values(by=column_name, inplace=True, ascending=False)
            return results_df
        else:
            return self.final_Y

    def get_metrics(self, metrics_function, key=None):
        return metrics_function(self.results, key)


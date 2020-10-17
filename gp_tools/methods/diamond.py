from collections import defaultdict
from gp_tools.methods.utils import init_value_by_seed
import scipy.stats
import numpy as np
import networkx as nx
import pandas as pd


class DIAMONnD:

    algorithm_name = 'DIAMONnD'

    @staticmethod
    def get_neighbors_and_degrees(G):

        neighbors, all_degrees = {}, {}
        for node in G.nodes():
            nn = set(G.neighbors(node))
            neighbors[node] = nn
            all_degrees[node] = G.degree(node)

        return neighbors, all_degrees

    @staticmethod
    def compute_all_gamma_ln(N):
        """
        precomputes all logarithmic gammas
        """
        gamma_ln = {}
        for i in range(1, N + 1):
            gamma_ln[i] = scipy.special.gammaln(i)

        return gamma_ln

    @staticmethod
    def reduce_not_in_cluster_nodes(all_degrees, neighbors, G, not_in_cluster, cluster_nodes, alpha):
        reduced_not_in_cluster = {}
        kb2k = defaultdict(dict)
        for node in not_in_cluster:

            k = all_degrees[node]
            kb = 0
            # Going through all neighbors and counting the number of module neighbors
            for neighbor in neighbors[node]:
                if neighbor in cluster_nodes:
                    kb += 1

            # adding wights to the the edges connected to seeds
            k += (alpha - 1) * kb
            kb += (alpha - 1) * kb
            kb2k[kb][k] = node

        # Going to choose the node with largest kb, given k
        k2kb = defaultdict(dict)
        for kb, k2node in kb2k.items():
            min_k = min(k2node.keys())
            node = k2node[min_k]
            k2kb[min_k][kb] = node

        for k, kb2node in k2kb.items():
            max_kb = max(kb2node.keys())
            node = kb2node[max_kb]
            reduced_not_in_cluster[node] = (max_kb, k)

        return reduced_not_in_cluster

    @staticmethod
    def pvalue(kb, k, N, s, gamma_ln):
        """
        -------------------------------------------------------------------
        Computes the p-value for a node that has kb out of k links to
        seeds, given that there's a total of s sees in a network of N nodes.

        p-val = \sum_{n=kb}^{k} HypergemetricPDF(n,k,N,s)
        -------------------------------------------------------------------
        """

        def logchoose(n, k, gamma_ln):
            if n - k + 1 <= 0:
                return scipy.infty
            lgn1 = gamma_ln[n + 1]
            lgk1 = gamma_ln[k + 1]
            lgnk1 = gamma_ln[n - k + 1]
            return lgn1 - [lgnk1 + lgk1]

        def gauss_hypergeom(x, r, b, n, gamma_ln):
            return np.exp(logchoose(r, x, gamma_ln) +
                          logchoose(b, n - x, gamma_ln) -
                          logchoose(r + b, n, gamma_ln))

        p = 0.0
        for n in range(kb, k + 1):
            if n > s:
                break
            prob = gauss_hypergeom(n, s, N - s, k, gamma_ln)
            # print prob
            p += prob

        if p > 1:
            return 1
        else:
            return p

    def diamond_iteration_of_first_X_nodes(self, G, S, X, alpha):

        """

        Parameters:
        ----------
        - G:     graph
        - S:     seeds
        - X:     the number of iterations, i.e only the first X gened will be
                 pulled in
        - alpha: seeds weight

        Returns:
        --------

        - added_nodes: ordered list of nodes in the order by which they
          are agglomerated. Each entry has 4 info:

          * name : dito
          * k    : degree of the node
          * kb   : number of +1 neighbors
          * p    : p-value at agglomeration

        """

        N = G.number_of_nodes()

        added_nodes = []

        # ------------------------------------------------------------------
        # Setting up dictionaries with all neighbor lists
        # and all degrees
        # ----------------------------   --------------------------------------
        neighbors, all_degrees = self.get_neighbors_and_degrees(G)

        # ------------------------------------------------------------------
        # Setting up initial set of nodes in cluster
        # ------------------------------------------------------------------

        cluster_nodes = set(S)
        not_in_cluster = set()
        s0 = len(cluster_nodes)

        s0 += (alpha - 1) * s0
        N += (alpha - 1) * s0

        # ------------------------------------------------------------------
        # precompute the logarithmic gamma functions
        # ------------------------------------------------------------------
        gamma_ln = self.compute_all_gamma_ln(N + 1)

        # ------------------------------------------------------------------
        # Setting initial set of nodes not in cluster
        # ------------------------------------------------------------------
        for node in cluster_nodes:
            not_in_cluster |= neighbors[node]
        not_in_cluster -= cluster_nodes

        # ------------------------------------------------------------------
        #
        # M A I N     L O O P
        #
        # ------------------------------------------------------------------

        all_p = {}

        while len(added_nodes) < X:

            # ------------------------------------------------------------------
            #
            # Going through all nodes that are not in the cluster yet and
            # record k, kb and p
            #
            # ------------------------------------------------------------------

            info = {}

            pmin = 10
            next_node = 'nix'
            reduced_not_in_cluster = self.reduce_not_in_cluster_nodes(all_degrees,
                                                                      neighbors, G,
                                                                      not_in_cluster,
                                                                      cluster_nodes, alpha)

            for node, kbk in reduced_not_in_cluster.items():
                # Getting the p-value of this kb,k
                # combination and save it in all_p, so computing it only once!
                kb, k = kbk
                try:
                    p = all_p[(k, kb, s0)]
                except KeyError:
                    p = self.pvalue(kb, k, N, s0, gamma_ln)
                    all_p[(k, kb, s0)] = p

                # recording the node with smallest p-value
                if p < pmin:
                    pmin = p
                    next_node = node

                info[node] = (k, kb, p)

            # ---------------------------------------------------------------------
            # Adding node with smallest p-value to the list of aaglomerated nodes
            # ---------------------------------------------------------------------
            added_nodes.append((next_node,
                                info[next_node][0],
                                info[next_node][1],
                                info[next_node][2]))

            # Updating the list of cluster nodes and s0
            cluster_nodes.add(next_node)
            s0 = len(cluster_nodes)
            not_in_cluster |= (neighbors[next_node] - cluster_nodes)
            not_in_cluster.remove(next_node)

        return added_nodes

    def __init__(self,
                 max_number_of_added_nodes=500,
                 alpha=1):
        self.max_number_of_added_nodes = max_number_of_added_nodes
        self.alpha = alpha
        self.node_final_values = dict()
        self.results = None

    def copy(self):
        return DIAMONnD(**self.get_params())

    def set_params(self,
                   max_number_of_added_nodes=500,
                   alpha=1):
        self.max_number_of_added_nodes = max_number_of_added_nodes
        self.alpha = alpha

    def get_params(self):

        params = {
            'max_number_of_added_nodes': self.max_number_of_added_nodes,
            'alpha': self.alpha
        }
        return params

    def run(self, G, seed_nodes):

        all_genes_in_network = set(G.nodes())
        seed_genes = set(seed_nodes)
        disease_genes = seed_genes & all_genes_in_network

        if len(disease_genes) != len(seed_genes):
            print("DIAMOnD(): ignoring    %s of %s seed genes that are not in the network" % (
                len(seed_genes - all_genes_in_network), len(seed_genes)))

        added_nodes = self.diamond_iteration_of_first_X_nodes(G,
                                                              disease_genes,
                                                              self.max_number_of_added_nodes,
                                                              self.alpha)

        added_nodes = [node for node, x, y, z in added_nodes]
        added_node_values = [1/(float(rank)+1) for rank, node in enumerate(added_nodes)]

        self.results = init_value_by_seed(G, added_nodes, added_node_values)

    def get_results_df(self, sorting=True, column_name='rw_value'):
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        results_df.columns = [column_name]
        if sorting:
            results_df.sort_values(by=column_name, inplace=True, ascending=False)
        return results_df

    def get_metrics(self, metrics_function, key=None):
        return metrics_function(self.results, key)


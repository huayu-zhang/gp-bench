import networkx as nx
import multiprocessing as mp


class SpParallel:

    """
    G = nx.fast_gnp_random_graph(1000, 0.05)
    nx_para = NetworkxParallel(G)
    nx_para.compute_diameter()
    """

    counter = mp.Value('i', 0)

    def __init__(self, G, n_jobs=None):

        self.G = G
        if not nx.is_connected(self.G):
            raise ValueError('Input graph is not connected!')
        self.G_number_of_nodes = self.G.number_of_nodes()
        if n_jobs is None:
            n_jobs = mp.cpu_count()
        self.n_jobs = n_jobs
        self._spl_dict = None
        self._eccentricity = None
        self._diameter = None
        self._closeness_centrality = None

    def reset_counter(self):

        self.counter.value = 0

    def single_source_spl(self, G, source):

        with self.counter.get_lock():
            self.counter.value += 1
            print('Node finished: (%s/%s)' % (self.counter.value, self.G_number_of_nodes), end='\r')
        return nx.single_source_shortest_path_length(G, source)

    def compute_pairwise_spl(self):

        self.reset_counter()
        print('Start computing pairwise shortest paths and lengths: ')

        map_generator = ((self.G, node) for node in self.G.nodes)

        pool = mp.Pool(self.n_jobs)
        spl = pool.starmap(self.single_source_spl, map_generator)
        pool.close()

        print('')
        print('Finished computing pairwise shortest path length. ')

        self._spl_dict = {node: length for node, length in zip(self.G.nodes, spl)}

    def _require_spl(self):
        if self._spl_dict is None:
            print('Compute pairwise shortest path and length first: ')
            self.compute_pairwise_spl()

    def get_pairwise_spl(self):
        return self._spl_dict

    def eccentricity(self):

        self._require_spl()

        self._eccentricity = nx.eccentricity(self.G, sp=self._spl_dict)
        print('Eccentricity computed')
        return self._eccentricity

    def diameter(self):

        if self._eccentricity is None:
            self.eccentricity()

        self._diameter = nx.diameter(self.G, self._eccentricity)
        print('Diameter computed')

        return self._diameter

    def _closeness_centrality_wrap(self, key):
        with self.counter.get_lock():
            self.counter.value += 1
            print('Closeness centrality nodes finished: (%s/%s)' % (self.counter.value, self.G_number_of_nodes), end='\r')
        return (self.G_number_of_nodes - 1)/sum(self._spl_dict[key].values())

    def closeness_centrality(self):

        self._require_spl()
        self.reset_counter()

        pool = mp.Pool(self.n_jobs)
        closeness_list = pool.map(self._closeness_centrality_wrap, self.G.nodes)
        pool.close()

        self._closeness_centrality = {node: value for node, value in zip(self.G.nodes, closeness_list)}

        return self._closeness_centrality



from gp_tools.IO import *
import numpy as np
import os
import networkx as nx
from gp_tools.mappings import DisgenetMappings


class MimMiner:

    db_path = './gp_tools/db/raw/mimminer/'
    edgelist_path = './gp_tools/db/edgelist/mimminer/'

    def __init__(self):
        self.available_sources = os.listdir(self.db_path)
        self.available_source_files = {source: self.db_path + source for source in self.available_sources}
        self.data_raw = None
        self.data_filtered = None
        self.filter_dict = None
        self.edge_df = None
        self.utilities = None

    def import_source(self):

        self.data_raw = pd.read_csv(self.available_source_files[self.available_sources[0]],
                                    sep='\t', index_col=0, header=None)
        self.data_raw.index = [str(index) for index in self.data_raw.index]
        self.data_raw.columns = self.data_raw.index

    def set_filters(self,
                    mim_similarity=0.6):
        self.filter_dict = {
            'mim_similarity': mim_similarity
        }

    def filter(self):

        mimminer_mat = self.data_raw.to_numpy()
        mimminer_adj = np.subtract(mimminer_mat >= self.filter_dict['mim_similarity'], np.identity(mimminer_mat.shape[0]))
        mimminer_adj_df = pd.DataFrame(mimminer_adj, index=self.data_raw.index, columns=self.data_raw.columns)

        mimminer_Graph = nx.convert_matrix.from_pandas_adjacency(mimminer_adj_df)
        self.data_filtered = nx.convert_matrix.to_pandas_edgelist(mimminer_Graph)

    def mapping_id(self, id_to_map=None):

        if id_to_map is not None:
            mappings = DisgenetMappings()
            self.data_filtered = mappings.map(df=self.data_filtered,
                                              index_in_df='source', to_which_mapping=id_to_map)
            self.data_filtered = mappings.map(df=self.data_filtered,
                                              index_in_df='target', to_which_mapping=id_to_map)

    def make_edge_df(self,
                     source='source',
                     target='target',
                     edge_type='DD_mimminer',
                     weight=1):

        self.edge_df = self.data_filtered[[source, target]].copy()
        self.edge_df.dropna(inplace=True)
        self.edge_df['type'] = edge_type
        self.edge_df['weight'] = weight
        self.utilities = EdgeDfUtilities(edge_df=self.edge_df)

    def quick_start(self):
        self.import_source()
        self.set_filters()
        self.filter()
        self.make_edge_df()



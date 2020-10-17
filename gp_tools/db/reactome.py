from gp_tools.IO import *
import pandas as pd
import networkx as nx
import os
from gp_tools.db.gtrd import UniprotMappings


class Reactome_PPI:

    db_path = './gp_tools/db/raw/reactome/'
    edgelist_path = './gp_tools/db/edgelist/reactome/'

    def __init__(self):

        self.available_sources = ['reactome.homo_sapiens.interactions.tab-delimited.txt']
        self.available_source_files = {source: self.db_path + source for source in self.available_sources}
        self.data_raw = None
        self.data_filtered = None
        self.filter_dict = None
        self.edge_df = None
        self.utilities = None
        self.summary = None

    def import_source(self):

        self.data_raw = pd.read_csv(self.available_source_files[self.available_sources[0]], sep='\t')
        self.data_raw['valid_ensembl_id'] = self.data_raw['Interactor 1 Ensembl gene id'].apply(lambda x: x != '-') & \
            self.data_raw['Interactor 2 Ensembl gene id'].apply(lambda x: x != '-')
        self.data_raw['non_self_loop'] = self.data_raw['Interactor 1 Ensembl gene id'] != \
            self.data_raw['Interactor 2 Ensembl gene id']
        self.data_raw['Interactor 1 Ensembl gene id'] = self.data_raw['Interactor 1 Ensembl gene id'].apply(
            lambda x: x.replace('ENSEMBL:', ''))
        self.data_raw['Interactor 2 Ensembl gene id'] = self.data_raw['Interactor 2 Ensembl gene id'].apply(
            lambda x: x.replace('ENSEMBL:', ''))

    def set_filters(self,
                    extra_filters=None):

        self.filter_dict = {
            'valid_ensembl_id': lambda x: x.valid_ensembl_id,
            'non_self_loop': lambda x: x.non_self_loop
        }

        if extra_filters is not None:
            for key in extra_filters:
                self.filter_dict[key] = extra_filters[key]

    def filter(self):

        self.data_filtered = self.data_raw.copy()

        if self.filter_dict is not None:
            for key in self.filter_dict:
                self.data_filtered = self.data_filtered[self.filter_dict[key](self.data_filtered)]

    def mapping_id(self, other_id):

        mappings = UniprotMappings()
        self.data_filtered = mappings.map(df=self.data_filtered, index_in_df='# Interactor 1 uniprot id',
                                          to_which_mapping=other_id)
        self.data_filtered = mappings.map(df=self.data_filtered, index_in_df='Interactor 2 uniprot id',
                                          to_which_mapping=other_id)

    def make_edge_df(self,
                     source='Interactor 1 Ensembl gene id',
                     target='Interactor 2 Ensembl gene id',
                     edge_type='PPI_reactome',
                     weight=1):

        self.edge_df = self.data_filtered[[source, target]].copy()
        self.edge_df.dropna(inplace=True)
        self.edge_df = flatten_mapping(df=self.edge_df,
                                       sep='|')
        self.edge_df['type'] = edge_type
        self.edge_df['weight'] = weight
        self.utilities = EdgeDfUtilities(edge_df=self.edge_df)

    def quick_start(self):
        self.import_source()
        self.set_filters()
        self.filter()
        self.make_edge_df()

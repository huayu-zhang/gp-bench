from gp_tools.IO import *
import pandas as pd
import os
from gp_tools.mappings import DisgenetMappings


class HPO_GDA:

    db_path = './gp_tools/db/raw/hpo/'
    edgelist_path = './gp_tools/db/edgelist/hpo/'

    def __init__(self):
        self.available_sources = ['genes_to_diseases.txt']
        self.available_source_files = {source: self.db_path + source for source in self.available_sources}
        self.data_raw = None
        self.data_filtered = None
        self.filter_dict = None
        self.edge_df = None
        self.utilities = None

    def import_source(self):

        self.data_raw = pd.read_csv(self.available_source_files[self.available_sources[0]],
                                    sep='\t', comment='#', header=None)
        self.data_raw.columns = ['entrez_id', 'gene_symbol', 'disease_id']
        self.data_raw['disease_db'] = [item.split(':')[0] for item in self.data_raw.disease_id]
        self.data_raw.disease_id = [item.split(':')[1] for item in self.data_raw.disease_id]
        self.data_raw['ensembl_id'] = get_gene_id_by_name(self.data_raw.gene_symbol)

    def set_filters(self,
                    db={'OMIM'},
                    extra_filters=None):

        self.filter_dict = {
            'right_db': lambda x: x.disease_db.apply(func=lambda y: y in db)
        }

        if extra_filters is not None:
            for key in extra_filters:
                self.filter_dict[key] = extra_filters[key]

    def filter(self):

        self.data_filtered = self.data_raw.copy()

        if self.filter_dict is not None:
            for key in self.filter_dict:
                self.data_filtered = self.data_filtered[self.filter_dict[key](self.data_filtered)]

    def mapping_id(self, id_to_map=None):

        if id_to_map is not None:
            mappings = DisgenetMappings()
            self.data_filtered = mappings.map(df=self.data_filtered,
                                              index_in_df='disease_id', to_which_mapping=id_to_map)

    def make_edge_df(self,
                     source='ensembl_id',
                     target='disease_id',
                     edge_type='GDA_HPO',
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


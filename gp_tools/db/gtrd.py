from gp_tools.IO import *
from gp_tools.mappings import UniprotMappings
import pandas as pd
import os


class GTRD:

    db_path = './gp_tools/db/raw/gtrd/'
    edgelist_path = './gp_tools/db/edgelist/gtrd/'

    @staticmethod
    def read_chip(file):
        uniprot_name = file.split('/')[-1].split('.')[0]
        chip = pd.read_csv(file, sep='\t')
        chip['uniprot_name'] = uniprot_name
        return chip

    def __init__(self):
        self.available_sources = os.listdir(self.db_path)
        self.available_source_files = {source: self.db_path + source + '/' for source in self.available_sources}
        self.data_raw = None
        self.data_filtered = None
        self.filter_dict = None
        self.edge_df = None
        self.utilities = None

    def import_source(self, source):
        chip_list = [self.read_chip(self.available_source_files[source] + file)
                     for file in os.listdir(self.available_source_files[source])]
        self.data_raw = pd.concat(chip_list)

    def set_filters(self,
                    min_binding_per_gene=8,
                    extra_filters=None):

        self.filter_dict = {
            'min_binding_per_gene': lambda x: x.SiteCount >= min_binding_per_gene,
        }

        if extra_filters is not None:
            for key in extra_filters:
                self.filter_dict[key] = extra_filters[key]

    def filter(self):
        self.data_filtered = self.data_raw.copy()

        if self.filter_dict is not None:
            for key in self.filter_dict:
                self.data_filtered = self.data_filtered[self.filter_dict[key](self.data_filtered)]

    def map_to_other_id(self, other_id):
        mappings = UniprotMappings()
        self.data_filtered = mappings.map(df=self.data_filtered, index_in_df='uniprot_name', to_which_mapping=other_id)
        return self.data_filtered

    def make_edge_df(self,
                     source='ID',
                     target='Ensembl',
                     edge_type='PPI_GTRD_CHIPSEQ',
                     weight=1):
        self.edge_df = self.data_filtered[[source, target]].copy()
        self.edge_df.dropna(inplace=True)
        self.edge_df['type'] = edge_type
        self.edge_df['weight'] = weight
        self.utilities = EdgeDfUtilities(self.edge_df)

    def quick_start(self):
        self.import_source(source='genes promoter[-1000,+100]')
        self.set_filters()
        self.filter()
        self.map_to_other_id(other_id='Ensembl')
        self.make_edge_df()


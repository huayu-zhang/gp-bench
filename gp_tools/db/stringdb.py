from gp_tools.IO import *
import os
from pyensembl import EnsemblRelease


# String Database

class StringDB:

    db_path = './gp_tools/db/raw/stringdb/'
    edgelist_path = './gp_tools/db/edgelist/stringdb/'

    def __init__(self):
        self.available_sources = os.listdir(self.db_path)
        self.available_source_files = {source: self.db_path + source for source in self.available_sources}
        self.data_raw = None
        self.data_filtered = None
        self.filter_dict = None
        self.edge_df = None
        self.utilities = None

    def import_source(self,
                      which_source='string_db_exp100.txt',
                      sep='\t'):
        self.data_raw = pd.read_csv(self.available_source_files[which_source], sep=sep)
        self.data_raw.protein1 = [protein.split('.')[1] for protein in self.data_raw.protein1]
        self.data_raw.protein2 = [protein.split('.')[1] for protein in self.data_raw.protein2]

    def set_filters(self,
                    min_experiment=360,
                    extra_filters=None):
        self.filter_dict = {
            'min_experiment': lambda x: x.experiments >= min_experiment
        }

        if extra_filters is not None:
            for key in extra_filters:
                self.filter_dict[key] = extra_filters[key]

    def filter(self):

        self.data_filtered = self.data_raw.copy()

        if self.filter_dict is not None:
            for key in self.filter_dict:
                self.data_filtered = self.data_filtered[self.filter_dict[key](self.data_filtered)]

    def mapping_id(self):

        esb = EnsemblRelease(77)
        self.data_filtered['ensembl_id_1'] = [esb.gene_id_of_protein_id(protein)
                                              for protein in self.data_filtered.protein1]
        self.data_filtered['ensembl_id_2'] = [esb.gene_id_of_protein_id(protein)
                                              for protein in self.data_filtered.protein2]

    def make_edge_df(self,
                     source='ensembl_id_1',
                     target='ensembl_id_2',
                     edge_type='PPI_stringdb',
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
        self.mapping_id()
        self.make_edge_df()



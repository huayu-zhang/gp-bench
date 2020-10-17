from gp_tools.IO import *
import os


class HuRI:

    db_path = './gp_tools/db/raw/huri/'
    edgelist_path = './gp_tools/db/edgelist/huri/'

    def __init__(self):
        self.available_sources = os.listdir(self.db_path)
        self.available_sources.remove('LitBM-17.psi')
        self.available_source_files = {source: self.db_path + source for source in self.available_sources}
        self.data_raw = None
        self.data_filtered = None
        self.filter_dict = None
        self.edge_df = None
        self.utilities = None

    def import_source(self, which_source='HuRI.tsv'):
        if 'tsv' in which_source:
            self.data_raw = xsv_to_edge_df(self.available_source_files[which_source], sep='\t')
        elif 'psi' in which_source:
            self.data_raw = psi_to_edge_df(self.available_source_files[which_source])
        else:
            raise ValueError('Invalid source: must be .tsv or .psi file')

    def set_filter(self):
        pass

    def filter(self):
        self.data_filtered = self.data_raw.copy()

    def map_to_other_id(self, other_id=None):
        pass

    def make_edge_df(self,
                     source='source',
                     target='target',
                     edge_type='PPI_HURI_Y2H',
                     weight=1):
        self.edge_df = self.data_filtered[[source, target]].copy()
        self.edge_df.dropna(inplace=True)
        self.edge_df['type'] = edge_type
        self.edge_df['weight'] = weight
        self.utilities = EdgeDfUtilities(self.edge_df)

    def quick_start(self):
        self.import_source()
        self.set_filter()
        self.filter()
        self.map_to_other_id()
        self.make_edge_df()


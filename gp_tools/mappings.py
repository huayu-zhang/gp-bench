from gp_tools.IO import *


class UniprotMappings:

    @staticmethod
    def expand_mapping(df, sep='; '):

        col_1 = [[cell] for cell in df.iloc[:, 0]]
        col_2 = df.iloc[:, 1]

        col_2_expanded = [cell.split(sep) for cell in col_2]
        n_match = [len(cell) for cell in col_2_expanded]
        col_1_expanded = [cell * n for cell, n in zip(col_1, n_match)]

        col_1_flattened = [item for sublist in col_1_expanded for item in sublist]
        col_2_flattened = [item for sublist in col_2_expanded for item in sublist]

        return pd.DataFrame({df.columns[0]: col_1_flattened, df.columns[1]: col_2_flattened})

    def __init__(self):

        self.mappings = pd.read_csv('./gp_tools/db/raw/gtrd/HUMAN_9606_idmapping_selected.tab', sep='\t', header=None)
        headings = pd.read_csv('./gp_tools/db/raw/gtrd/uniprot_mapping_headings.csv', header=None)
        self.mappings.columns = headings[0]

        self.available_mappings = ['UniProtKB-AC -> ' + col for col in self.mappings.columns]
        self._available_mappings = self.mappings.columns

    def map(self, df, index_in_df, to_which_mapping):
        mapping_selected = flatten_mapping(df=self.mappings[['UniProtKB-AC', to_which_mapping]].dropna(),
                                           sep='; ',
                                           which_column='right')
        mapped_df = pd.merge(left=df, left_on=index_in_df, right=mapping_selected, right_on='UniProtKB-AC')

        return mapped_df


class DisgenetMappings:

    def __init__(self):
        self.mappings = pd.read_csv('./gp_tools/db/raw/disgenet/disease_mappings.tsv', sep='|')
        self.map_dict = {db: self.mappings[self.mappings.vocabulary == db][['diseaseId', 'code']]
                         for db in self.mappings.vocabulary.unique()}

        for key in self.map_dict:
            self.map_dict[key].columns = ['UMLS', key]

        self.available_mappings = ['UMLS -> ' + key for key in self.map_dict]
        self._available_mappings = self.map_dict.keys()

    def map(self, df, index_in_df, to_which_mapping):
        mapped_df = pd.merge(left=df, left_on=index_in_df, right=self.map_dict[to_which_mapping], right_on='UMLS')
        return mapped_df

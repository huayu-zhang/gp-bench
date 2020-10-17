import pandas as pd
import itertools
import networkx as nx
import re
import pickle
from pyensembl import EnsemblRelease


def pickle_dump_to_file(object, file_path):
    fileObject = open(file_path, 'wb')
    pickle.dump(object, fileObject)
    fileObject.close()


def df_rows_to_dict(df):
    """
    from df rows to dict containing {column name: value} pairs, return a list of dicts
    :param df:
    :return: list of dicts
    """
    list_of_attr = [dict(row._asdict()) for row in df.itertuples(index=False)]
    return list_of_attr


def df_to_edgelist(df, source=None, target=None, write_path=None):
    """
    Function to convert the first 2 columns of df into edge pairs, and take the rest of columns
    into attr of edge.

    :arg df: the dataframe
    :return: nx edgelist
    """
    if (source is None) and (target is None):
        nodes_1 = df.iloc[:, 0]
        nodes_2 = df.iloc[:, 1]
        attrs = df_rows_to_dict(df.iloc[:, 2:])
    else:
        nodes_1 = df[source]
        nodes_2 = df[target]
        attrs = df_rows_to_dict(df.drop(columns=[source, target]))

    list_of_edge = [(n1, n2, attr) for n1, n2, attr in zip(nodes_1, nodes_2, attrs)]
    G = nx.from_edgelist(list_of_edge)

    if write_path is not None:
        nx.write_edgelist(G, write_path)

    return nx.to_edgelist(G)


def psi_to_edge_df(path, regex='ENSG\d+', weight=1, type_edge=None):

    # read pattern by line
    pattern = re.compile(regex)
    f = open(path, 'r')
    edges = [pattern.findall(line)[:2] for line in f]
    f.close()

    edge_df = pd.DataFrame(edges, columns=['source', 'target'])

    # initialize edge attr: type and weight
    if type_edge is None:
        type_edge = path.split('/')[-1].split('.')[-2]
    edge_df['type'] = type_edge
    edge_df['weight'] = weight

    return edge_df


def xsv_to_edge_df(path, weight=1, type_edge=None, **kwargs):
    edge_df = pd.read_csv(path, **kwargs).iloc[:, :2]
    edge_df.columns = ['source', 'target']

    if type_edge is None:
        type_edge = path.split('/')[-1].split('.')[-2]
    edge_df['type'] = type_edge
    edge_df['weight'] = weight

    return edge_df


# ensembl annotations of genes

ensembl = EnsemblRelease(77)


def get_gene_id_by_name(gene_symbol_list):

    def _get_gene_id_by_name(gene_symbol):
        # The error tolerating function
        try:
            return ensembl.gene_ids_of_gene_name(gene_symbol)[0]
        except ValueError:
            return None

    if isinstance(gene_symbol_list, str):
        return _get_gene_id_by_name(gene_symbol_list)
    else:
        return [_get_gene_id_by_name(gene) for gene in gene_symbol_list]


def get_gene_name_by_ensembl_id(ensembl_id_list):

    def _get_gene_name_by_ensembl_id(ensembl_id):
        try:
            return ensembl.gene_name_of_gene_id(ensembl_id)
        except ValueError:
            return None

    if isinstance(ensembl_id_list, str):
        return _get_gene_name_by_ensembl_id(ensembl_id_list)
    else:
        return [_get_gene_name_by_ensembl_id(id) for id in ensembl_id_list]


# Flattening multi_to_multi relationships


def flatten_mapping(df, sep, which_column='both'):
    """
    Flatten DataFrame with multi to multi relationship separated by sep
    First 2 columns are used. The left, right or both columns can be flattened
    :param df: DataFrame in which first 2 columns are to be flattened
    :param sep: separator used for multi mappings
    :param which_column: {'left', 'right', 'both'}
    :return: Flatten DataFrame with 2 columns
    """
    def _flatten_mapping(df, sep=sep, on_left=True):

        if on_left:
            as_ref = df.iloc[:, 1]
            to_flatten = df.iloc[:, 0]
        else:
            as_ref = df.iloc[:, 0]
            to_flatten = df.iloc[:, 1]

        as_ref = [[item] for item in as_ref]

        to_flatten_splitted = [item.split(sep) for item in to_flatten]
        n_match = [len(item) for item in to_flatten_splitted]
        as_ref_multi = [cell * n for cell, n in zip(as_ref, n_match)]

        as_ref_flattened = [item for sublist in as_ref_multi for item in sublist]
        to_expand_flattened = [item for sublist in to_flatten_splitted for item in sublist]

        if on_left:
            return pd.DataFrame({df.columns[0]: to_expand_flattened, df.columns[1]: as_ref_flattened})
        else:
            return pd.DataFrame({df.columns[0]: as_ref_flattened, df.columns[1]: to_expand_flattened})

    if which_column in {'both', 'left'}:
        df = _flatten_mapping(df, on_left=True)

    if which_column in {'both', 'right'}:
        df = _flatten_mapping(df, on_left=False)

    return df


# Functionality of db classes

class EdgeDfUtilities:

    def __init__(self, edge_df):
        self.edge_df = edge_df

    def get_edge_df(self):
        return self.edge_df

    def to_edgelist(self,
                    path=None
                    ):
        edgelist = df_to_edgelist(self.edge_df, write_path=path)
        print('Filtered edge list saved at %s' % path)
        return edgelist

    def to_csv(self, path):
        self.edge_df.to_csv(path)
        print('Edge dataframe saved at %s' % path)

    def dump(self, path):
        pickle_dump_to_file(self.edge_df, path)
        print('Edge dataframe dumped at %s' % path)

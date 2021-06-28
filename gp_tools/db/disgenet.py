from gp_tools.IO import *
from gp_tools.mappings import DisgenetMappings
import rdflib
import sqlite3
import pandas as pd
import numpy as np
import os


# methods for processing the gda table

class DisGeNET_GDA:

    db_path = './gp_tools/db/raw/disgenet/'
    edgelist_path = './gp_tools/db/edgelist/disgenet/'

    @staticmethod
    def is_prevalent_elements(series, k):
        index_prevalent = set(
            series.value_counts()[series.value_counts() >= k].index)
        is_prevalent = series.apply(func=lambda x: x in index_prevalent)
        return is_prevalent

    def __init__(self):

        self.available_sources = ['all_gene_disease_associations.tsv']
        self.available_source_files = {source: self.db_path + source for source in self.available_sources}
        self.data_raw = None
        self.data_filtered = None
        self.filter_dict = None
        self.edge_df = None
        self.utilities = None
        self.summary = None

    def import_source(self, which_source, sep='\t'):

        self.data_raw = pd.read_csv(self.available_source_files[which_source], sep=sep)
        self.data_raw['disease_class_set'] = self.data_raw.diseaseClass.apply(lambda x: set(str(x).split(';')))
        self.data_raw['source_set'] = self.data_raw.source.apply(lambda x: set(str(x).split(';')))
        self.data_raw['ensembl_id'] = get_gene_id_by_name(self.data_raw.geneSymbol)

    def set_filters(self,
                    mim_score=0.1,
                    exclude_source={'CTD_mouse', 'CTD_rat', 'RGD', 'MGD'},
                    exclude_type={'group', 'phenotype'},
                    min_gene_per_disease=0,
                    extra_filters=None):

        self.filter_dict = {
            'min_score': lambda x: x.score >= mim_score,
            'exclude_source': lambda x: x.source_set.apply(func=lambda y: not y.intersection(exclude_source)),
            'exclude_type': lambda x: x.diseaseType.apply(func=lambda y: y not in exclude_type),
            'min_gene_per_disease': lambda x: self.is_prevalent_elements(x.diseaseType, k=min_gene_per_disease)
        }

        if extra_filters is not None:
            for key in extra_filters:
                self.filter_dict[key] = extra_filters[key]

    def filter(self):

        self.data_filtered = self.data_raw.copy()

        if self.filter_dict is not None:
            for key in self.filter_dict:
                self.data_filtered = self.data_filtered[self.filter_dict[key](self.data_filtered)]

    def mapping_id(self, id_to_map):

        mappings = DisgenetMappings()
        self.data_filtered = mappings.map(df=self.data_filtered, index_in_df='diseaseId', to_which_mapping=id_to_map)

    def make_edge_df(self,
                     source='ensembl_id',
                     target='diseaseId',
                     edge_type='GDA_DisGeNet',
                     weight=1):

        if edge_type in self.data_filtered.columns:
            self.edge_df = self.data_filtered[[source, target, edge_type]].copy()
            self.edge_df.dropna(inplace=True)
        else:
            self.edge_df = self.data_filtered[[source, target]].copy()
            self.edge_df.dropna(inplace=True)
            self.edge_df['type'] = edge_type
            
        self.edge_df['weight'] = weight
        self.utilities = EdgeDfUtilities(edge_df=self.edge_df)

    def gda_summary(self):

        self.summary = {
            'numbesb.gener_association': len(self.data_filtered.index),
            'disease_classes': set.union(*self.data_filtered.disease_class_set),
            'sources': set.union(*self.data_filtered.source_set),
            'numeric_summary': self.data_filtered.describe(),
            'categorical_summary': self.data_filtered.describe(include=np.object)
        }

        self.summary['categorical_summary'].disease_class_set['unique'] = set.union(*self.data_filtered.disease_class_set).__len__()
        self.summary['categorical_summary'].source_set['unique'] = set.union(*self.data_filtered.source_set).__len__()

        return self.summary

    def get_summary(self):
        return self.summary


# methods to query sqlite database of disgenet


class DisGeNET_sqlite:

    db_path = './gp_tools/db/raw/disgenet/disgenet_2018.db'

    def __init__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cur = None

    def get_table_names(self):
        self.cur = self.conn.cursor()
        self.cur.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            ORDER BY name;
            """
        )
        table_names = [list(name)[0] for name in self.cur.fetchall()]
        self.cur.close()
        return table_names

    def get_column_names_of_table(self, table_name):
        self.cur = self.conn.cursor()
        self.cur.execute(
            """
            PRAGMA table_info('%s')
            """ % table_name
        )
        column_names = [list(name)[1] for name in self.cur.fetchall()]
        self.cur.close()
        return column_names

    def get_table(self, table_name):
        self.cur = self.conn.cursor()
        self.cur.execute(
            """
            SELECT *
            FROM %s;
            """ % table_name
        )
        table = pd.DataFrame.from_records(self.cur.fetchall(),
                                          columns=self.get_column_names_of_table(table_name))
        self.cur.close()
        return table

    def get_query(self, sql):
        self.cur = self.conn.cursor()
        self.cur.execute(sql)
        query = self.cur.fetchall()
        self.cur.close()
        return query

    def close_connection(self):
        self.conn.close()


# method to query RDF linked library


class DisGeNET_rdf:

    rdf_path = './gp_tools/db/raw/digenet/rdf/'

    @staticmethod
    def triples_to_list(rdf_graph):
        return [(s, p, o) for s, p, o in rdf_graph]

    def __init__(self):
        self.rdf_files = [self.rdf_path + file for file in os.listdir(self.rdf_path)]
        self.available_sources = os.listdir(self.rdf_path)
        self._g = rdflib.Graph()
        self._format_files = 'turtle'

    def load_source(self, source):
        self._g.load(self.rdf_path + source)

    def get_triples(self):
        return self.triples_to_list(self._g)

    def get_query(self, sparql):
        query = self._g.query(sparql)
        return self.triples_to_list(query)

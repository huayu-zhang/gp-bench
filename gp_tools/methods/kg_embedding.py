import networkx as nx
import pandas as pd
import numpy as np
import os
import csv
from scipy import spatial


class DglKE:

    algorithm_name = 'KGEmbedding'

    @staticmethod
    def cosine_sim(v1, v2):
        return 1 - spatial.distance.cosine(v1, v2)

    def __init__(self,
                 params=None,
                 emb_path='./ckpts/',
                 delimiter='\t',
                 protein_identifier='ENS',
                 entity_file=None,
                 relation_file=None
                 ):

        if params is None:
            params = {
                'dataset': 'PPI_KE',
                'data_path': './ke_train/',
                'data_files': 'train.tsv valid.tsv test.tsv',
                'format': 'raw_udd_htr',
                'model_name': 'TransE_l2',
                'batch_size': 2048,
                'neg_sample_size': 256,
                'hidden_dim': 400,
                'gamma': 12,
                'lr': 0.1,
                'max_step': 100000,
                'log_interval': 1000,
                'batch_size_eval': 16,
                'regularization_coef': 1.00E-07,
                'neg_sample_size_eval': 10000}

        self.params = params
        self.emb_path = emb_path
        self.delimiter = delimiter
        self.protein_identifier = protein_identifier
        self.entity_file = entity_file
        self.relation_file = relation_file
        self.node_emb = None
        self.relation_emb = None
        self.entity2id = {}
        self.rel2id = {}
        self.seed2id = {}
        self.embeddings = {}
        self.results = {}
        self.run_id = -1

    def copy(self):
        return DglKE(params=self.params,
                     protein_identifier=self.protein_identifier)

    def get_params(self):
        return self.params

    def set_params(self, params=None):
        self.params = params

    def kge_id_mapping(self, seed_nodes):
        with open(self.entity_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['id', 'entity'])
            for row_val in reader:
                id = row_val['id']
                self.entity2id[row_val['entity']] = int(id)

        with open(self.relation_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['id', 'entity'])
            for row_val in reader:
                id = row_val['id']
                self.rel2id[row_val['entity']] = int(id)

        for seed in seed_nodes:
            try:
                self.seed2id[seed] = self.entity2id[seed]
            except:
                self.seed2id[seed] = None

    def run(self, G,
            seed_nodes=None, cv=None):
        """
        :param G: Input graph in Networkx Format
        :param seed_nodes: Seeds for RWR algorithm
        :param cv: cross validation iterator (cv_[node])
        """
        if cv != None:
            self.params['data_path'] = self.params['data_path']+cv+"/"

        if not os.path.exists(self.params['data_path']):
            os.mkdir(self.params['data_path'])

        # Here convert networkx to the format used for dgl-ke
        # create an edgelist dataset
        triples = list(nx.generate_edgelist(G, delimiter=self.delimiter))
        num_triples = len(triples)

        # split into train, validate & test sets and save
        idx = np.arange(num_triples)
        np.random.shuffle(idx)

        train_count = int(num_triples * 0.9)
        valid_count = int(num_triples * 0.05)
        train_set = idx[:train_count].tolist()
        valid_set = idx[train_count:train_count+valid_count].tolist()
        test_set = idx[train_count+valid_count:].tolist()

        with open(os.path.join(self.params['data_path'], self.params['data_files'].split(" ")[0]), 'w+') as f:
            for id in train_set:
                f.writelines(triples[id] + "\n")

        with open(os.path.join(self.params['data_path'], self.params['data_files'].split(" ")[1]), 'w+') as f:
            for id in valid_set:
                f.writelines(triples[id] + "\n")

        with open(os.path.join(self.params['data_path'], self.params['data_files'].split(" ")[2]), 'w+') as f:
            for id in test_set:
                f.writelines(triples[id] + "\n")

        # Here invoke the command line used for running the algorithm
        self.run_id += 1
        os.system('dglke_train --dataset {} --data_path {} --data_files {} --format "{}" --model_name {} --batch_size {} \
        --neg_sample_size {} --hidden_dim {} --gamma {} --lr {} --max_step {} --log_interval {} --batch_size_eval {} -adv \
        --regularization_coef {} --test --num_thread 1 --gpu 0 --num_proc 0 --neg_sample_size_eval {} '.format(*list(self.params.values())))

        # Training generates the following files: (xxx=<dataset>_<model_name>_)
        # Entity embedding: ./ckpts/<model_name>_<dataset>_<run_id>/xxx_entity.npy
        # Relation embedding: ./ckpts/<model_name>_<dataset>_<run_id>/xxx_relation.npy
        # The entity id mapping, formated in <entity_name> <entity_id> pair: <data_path>/entities.tsv
        # The relation id mapping, formated in <relation_name> <relation_id> pair: <data_path>/relations.tsv

        # if self.entity_file == None:
        self.entity_file = os.path.join(self.params['data_path'], "entities.tsv")
        # if self.relation_file == None:
        self.relation_file = os.path.join(self.params['data_path'], "relations.tsv")

        self.node_emb = os.path.join(self.emb_path, self.params['model_name']+"_"+self.params['dataset']+"_"+str(self.run_id)+"/"+self.params['dataset']+"_"+self.params['model_name']+"_entity.npy")
        self.relation_emb = os.path.join(self.emb_path, self.params['model_name']+"_"+self.params['dataset']+"_"+str(self.run_id)+"/"+self.params['dataset']+"_"+self.params['model_name']+"_relation.npy")

        # create mapping dictionaries to KGE ids assigned by training algorithm
        self.kge_id_mapping(seed_nodes)

        # Some process here to convert algorithm output to dict of embeddings {node1: v1, node2: v2...}

        self.embeddings = np.load(self.node_emb)

        # Calculate similarity score
        if seed_nodes is not None:
            for node, keid in self.entity2id.items():
                self.results[node] = max([
                    self.cosine_sim(self.embeddings[keid], self.embeddings[seed_keid]) for seed_keid in self.seed2id.values()])

    def get_results_df(self, sorting=True, column_name='max_sim'):
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        results_df.columns = [column_name]
        if sorting:
            results_df.sort_values(by=column_name, inplace=True, ascending=False)
        return results_df

    def get_metrics(self, metrics_function, key=None):
        return metrics_function(self.results, key)

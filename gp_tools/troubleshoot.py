from gp_tools.methods import *
from gp_tools.IO import *
import networkx as nx
import pandas as pd
import numpy as np
import random
from collections import namedtuple
import multiprocessing as mp
import itertools
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import pickle
import sqlite3
import rdflib
import pyensembl
import os


# ILDP pilot

# UMLS to OMIM mapping
kg = rdflib.Graph()
kg.parse('./data/disgenet/ls-umls2omim.ttl', format='turtle')

result_s = [str(s).split('/')[-1] for s, p, o in kg]
result_o = [str(o).split('/')[-1] for s, p, o in kg]

umls_omim_map = pd.DataFrame(result_s, columns=['umls'])
umls_omim_map['omim'] = result_o
umls_omim_map.to_csv('./data/disgenet/umls2omim_map_df.csv')

# DD
mimminer_sim = pd.read_csv('./data/mimminer/MimMiner_Exp_AC_T_TXCS_basedonACMESH_filt_RW.mat',
                           sep='\t', index_col=0, header=None)
mimminer_sim.index = [str(index) for index in mimminer_sim.index]
mimminer_sim.columns = mimminer_sim.index

mimminer_mat = mimminer_sim.to_numpy()
mimminer_adj = np.subtract(mimminer_mat, np.identity(mimminer_mat.shape[0]))

mimminer_adj_df = pd.DataFrame(mimminer_adj, index=mimminer_sim.index, columns=mimminer_sim.columns)

S2 = np.diag(np.sqrt(mimminer_mat.sum(axis=0))) @ mimminer_mat @ np.diag(np.sqrt(mimminer_mat.sum(axis=0)))

# PD

PD = pd.read_csv('./data/disgenet/gda_selected.csv', index_col=0)
PD = pd.merge(left=PD, left_on='diseaseId', right=umls_omim_map, right_on='umls')

PD['in_DD'] = [omim in mimminer_adj_df.index for omim in PD.omim]

PD = PD[PD.in_DD]
PD = PD[['geneSymbol', 'omim']]

# PP

PP = pd.read_csv('./data/PPI/PP_small.csv', index_col=0)


DD_df = pd.DataFrame(DD.edges)
DD_df.columns = ['omim_1', 'omim_2']

DD_df = pd.merge(left=DD_df, right=umls_omim_map, left_on='omim_1', right_on='omim')
DD_df.drop(columns=['omim'], inplace=True)
DD_df.columns = ['omim_1', 'omim_2', 'umls_1']

DD_df = pd.merge(left=DD_df, right=umls_omim_map, left_on='omim_2', right_on='omim')
DD_df.drop(columns=['omim'], inplace=True)
DD_df.columns = ['omim_1', 'omim_2', 'umls_1', 'umls_2']

DD_df.drop(columns=['omim_1', 'omim_2'], inplace=True)
DD_df.to_csv('./data/mimminer/dd_df.csv')





def seed_nodes_in_graph(PP, seed_nodes):
    node_set = set(PP.iloc[:, 0]).union(set(PP.iloc[:, 1]))

    in_graph_seeds = [seed for seed in seed_nodes if seed in node_set]

    return in_graph_seeds


def seed_interaction(PP, seed_nodes):
    PP_copy = PP.copy()
    seed_nodes = set(seed_nodes)
    PP_copy['gene_1_in_seed'] = [gene in seed_nodes for gene in PP.iloc[:, 0]]
    PP_copy['gene_2_in_seed'] = [gene in seed_nodes for gene in PP.iloc[:, 1]]

    return PP[PP_copy.gene_1_in_seed | PP_copy.gene_2_in_seed]


# Data import
PPI_triples = pd.read_csv('./data/example_graphs/barabasi_PPI_format_adjusted.TSV',
                          sep='\t', comment='#')
PPI_triples['weight'] = 1
PPI_triples.gene_symbol_1 = [str(symbol) for symbol in PPI_triples.gene_symbol_1]
PPI_triples.gene_symbol_2 = [str(symbol) for symbol in PPI_triples.gene_symbol_2]

PPI_edges = df_to_edgelist(PPI_triples.iloc[:, 2:])

# PPI
PP = nx.Graph()
PP.add_edges_from(PPI_edges)
for u, v in PP.edges:
    if u == v:
        PP.remove_edge(u, v)

# ensembl release 77
esb = pyensembl.EnsemblRelease(77)
esb.gene_by_id('ENSG00000148143')




# Selected PPI
y2h_ht14 = pd.read_csv('./data/PPI/HI-II-14_trim.csv', sep='\t', header=None)
y2h_ht14.columns = ['uniprot_id_1', 'uniprot_id_2', 'ensembl_anno_1', 'ensembl_anno_2']
y2h_ht14['ensembl_gene_1'] = [anno.split('|')[-1].split('.')[0].replace('ensembl:', '')
                              for anno in y2h_ht14.ensembl_anno_1]
y2h_ht14['ensembl_gene_2'] = [anno.split('|')[-1].split('.')[0].replace('ensembl:', '')
                              for anno in y2h_ht14.ensembl_anno_2]


def gene_name_of_gene_id(gene_id):
    try:
        return esb.gene_name_of_gene_id(gene_id)
    except ValueError:
        return None


y2h_ht14['gene_symbol_1'] = [gene_name_of_gene_id(gene) for gene in y2h_ht14.ensembl_gene_1]
y2h_ht14['gene_symbol_2'] = [gene_name_of_gene_id(gene) for gene in y2h_ht14.ensembl_gene_2]

y2h_ht14.dropna(inplace=True)
y2h_ht14[['gene_symbol_1', 'gene_symbol_2']].to_csv('./data/PPI/HI-II-14_df.csv')


# test: sum([list(gene).__len__() != 15 for gene in y2h_ht14.ensembl_gene_1])
# test: sum(['ENSG' not in gene for gene in y2h_ht14.ensembl_gene_1])


# PMID: 21900206




seed_interaction(string_db_df, seed_genes.gene_symbol)

seed_nodes_in_graph(string_db_df, seed_genes.gene_symbol)


# TF binding

def read_chip(file):
    uniprot_name = file.split('/')[-1].split('.')[0]
    chip = pd.read_csv(file, sep='\t')
    chip['uniprot_name'] = uniprot_name
    return chip


chip_list = [read_chip('./data/PPI/genes promoter[-1000,+100]/' + file) for file in os.listdir('./data/PPI/genes promoter[-1000,+100]')]

chip_df = pd.concat(chip_list)

chip_df_selected = chip_df[chip_df.SiteCount >= 8]

chip_df_selected.to_csv('./data/PPI/chip_df_selected.csv')

PPI_y2h = pd.read_csv('./data/PPI/HI-II-14_df.csv', index_col=0)
PPI_reg = pd.read_csv('./data/PPI/PMID21900206_df.csv')
PPI_TF = pd.read_csv('./data/PPI/chip_df_anno.csv', index_col=0)

PP = pd.concat([PPI_y2h, PPI_reg, PPI_TF])

PP.drop_duplicates(inplace=True)
PP.to_csv('./data/PPI/PPI_selected_df.csv')

# StringGO

string_db = pd.read_csv('./data/PPI/9606.protein.links.full.v11.0.txt', sep=' ')
string_db.experiments.describe()

string_db_df = string_db[string_db.experiments > 360]

string_db_df = string_db_df[['protein1', 'protein2']]
string_db_df.protein1 = [ensembl.split('.')[1] for ensembl in string_db_df.protein1]
string_db_df.protein2 = [ensembl.split('.')[1] for ensembl in string_db_df.protein2]

string_db_df['gene_symbol_1'] = [esb.gene_by_protein_id(ensembl).gene_name for ensembl in string_db_df.protein1]
string_db_df['gene_symbol_2'] = [esb.gene_by_protein_id(ensembl).gene_name for ensembl in string_db_df.protein2]

string_db_df.drop(columns=['protein1', 'protein2'], inplace=True)

PP = pd.read_csv('./data/PPI/PPI_selected_df.csv', index_col=0)

PP_small_plus = pd.concat([PP, string_db_df])
PP_small_plus.drop_duplicates(inplace=True)
PP_small_plus.to_csv('./data/PPI/PP_small.csv')


PP_medium = string_db[string_db.experiments > 400]

PP_medium = PP_medium[['protein1', 'protein2']]
PP_medium.protein1 = [ensembl.split('.')[1] for ensembl in PP_medium.protein1]
PP_medium.protein2 = [ensembl.split('.')[1] for ensembl in PP_medium.protein2]

PP_medium['gene_symbol_1'] = [esb.gene_by_protein_id(ensembl).gene_name for ensembl in PP_medium.protein1]
PP_medium['gene_symbol_2'] = [esb.gene_by_protein_id(ensembl).gene_name for ensembl in PP_medium.protein2]

PP_medium.drop(columns=['protein1', 'protein2'], inplace=True)
PP_medium = pd.concat([PP, PP_medium, seed_interaction(string_db_df, seed_genes.gene_symbol)])
PP_medium.drop_duplicates(inplace=True)
PP_medium.to_csv('./data/PPI/PP_medium_plus.csv')


PP_large = pd.concat([PP, string_db_df])
PP_large.drop_duplicates(inplace=True)
PP_large.to_csv('./data/PPI/PP_large.csv')


# PD
# Disgenet diseaseId is the UMLS-CUI index
dgn = DisGeNET()
dgn.gda_import('./data/disgenet/all_gene_disease_associations.tsv')
dgn.gda_filter()

dgn.gda_filtered.to_csv('./data/disgenet/gda_selected.csv')

PD = dgn.gda_filtered[['geneSymbol', 'diseaseId']]

# DD
mimminer_sim = pd.read_csv('./data/mimminer/MimMiner_Exp_AC_T_TXCS_basedonACMESH_filt_RW.mat',
                           sep='\t', index_col=0, header=None)
mimminer_sim.index = [str(index) for index in mimminer_sim.index]
mimminer_sim.columns = mimminer_sim.index

mimminer_mat = mimminer_sim.to_numpy()
mimminer_adj = np.subtract(mimminer_mat >= 0.6, np.identity(mimminer_mat.shape[0]))

mimminer_adj_df = pd.DataFrame(mimminer_adj, index=mimminer_sim.index, columns=mimminer_sim.columns)

DD = nx.convert_matrix.from_pandas_adjacency(mimminer_adj_df)

DD_df = pd.DataFrame(DD.edges)
DD_df.columns = ['omim_1', 'omim_2']

DD_df = pd.merge(left=DD_df, right=umls_omim_map, left_on='omim_1', right_on='omim')
DD_df.drop(columns=['omim'], inplace=True)
DD_df.columns = ['omim_1', 'omim_2', 'umls_1']

DD_df = pd.merge(left=DD_df, right=umls_omim_map, left_on='omim_2', right_on='omim')
DD_df.drop(columns=['omim'], inplace=True)
DD_df.columns = ['omim_1', 'omim_2', 'umls_1', 'umls_2']

DD_df.drop(columns=['omim_1', 'omim_2'], inplace=True)
DD_df.to_csv('./data/mimminer/dd_df.csv')

G = nx.Graph()
G.add_edges_from(df_to_edgelist(PP))
G.add_edges_from(df_to_edgelist(PD))
G.add_edges_from(df_to_edgelist(DD_df))

[nodes.__len__() for nodes in list(nx.connected_components(G))]


# List of seed
seed_genes = pd.read_csv('./data/example_graphs//seed_genes.tsv', delimiter='\t')
seed_nodes = set(seed_genes.iloc[:, 0]).intersection(set(G.nodes))
seed_nodes = list(seed_nodes)


loo = LOOCV(base_algorithm=RandomWalk(), n_jobs=8)

loo.cv(G, seed_nodes)




# Disgenet
dgn = DisGeNET()
dgn.gda_import('./data/disgenet/all_gene_disease_associations.tsv')
dgn.gda_filter()

dgn.gda_summary()


# Time simulation network
TSN = TimeSimulationNetwork(nx.closeness_centrality, n_nodes=[300, 1000, 3000])

TSN.time_it()


cc = [G.subgraph(c).copy() for c in nx.connected_components(G)]

largest_C = G.subgraph(max(nx.connected_components(G), key=len))

nx.radius(largest_C)



# Graph metrics pass

gm = GraphMetrics(G)
gm.compute_metrics()


# Modularity analysis (PASS)
MA = ModularityAnalysis(G)
MA.compute_communities()
MA.compute_modularity()


# Pairwise shortest path (pass)
PSM = PairwiseMetrics(G, seed_nodes)
PSM.compute_metrics()
PSM.get_heatmaps()


# Basic node metrics
NM = NodeMetrics(G, seed_nodes)
NM.compute_fast_metrics()
NM.get_metrics_df()
NM.visualize_metrics()
NM.visualize_metrics(ylim=(0, 0.005))
NM.get_metrics_summary()

degree_centrality = nx.degree_centrality(G)
ev_centrality = nx.eigenvector_centrality(G)


nm_df = pd.DataFrame.from_dict(degree_centrality, orient='index', columns=['y'])


def seed_one_hot(G, seed_nodes):
    OH_dict = dict(zip(G.nodes, [0] * len(G.nodes)))
    for seed_node in seed_nodes:
        OH_dict[seed_node] = 1
    return OH_dict


seed_OH_dict = seed_one_hot(G, seed_nodes)

nm_df['is_seed'] = seed_OH_dict.values()
nm_df['x'] = 'degree_centrality'

nm_df_melt = nm_df.melt(id_vars='is_seed')

plt.close()
im = sns.boxplot(x='x', y='y', color='grey', data=nm_df)
plt.ylim(0, 0.005)
plt.show()


# seed_genes = pd.read_csv('./data/example_graphs/seed_genes.tsv', delimiter='\t')
# seed_nodes = list(seed_genes.gene_id)


# Id to gene name mapping
PPI_map = PPI_triples.iloc[:, [0, 2]]
PPI_map.drop_duplicates(inplace=True)
PPI_map.index = PPI_map['gene_ID_1']

N2V = Node2Vec(n_jobs=6)
N2V.setup(G)

LOO = LOOCV(base_algorithm=N2V)
LOO.cv(G, seed_nodes)
LOO.get_results_df().to_csv('N2V_LOO.csv')


N2V.run(G, seed_nodes)
N2V.df_sim.to_csv('N2V_sim.csv')


# 2D presentation of graph nodes

N2V.dimensions = 10
N2V.embedding_results = N2V.learn_embeddings()

embeddings = np.array([N2V.embedding_results.wv.get_vector(node) for node in G.nodes])

pca = PCA(n_components=2)
embedding_2D = pca.fit_transform(X=embeddings)

x = [i[0] for i in embedding_2D]
y = [i[1] for i in embedding_2D]

rw_result_df_list = list()

for max_iter in range(10):
    RW = RandomWalk()
    RW.max_iter = max_iter
    RW.run(G, seed_nodes)
    rw_result_df_list.append(RW.get_results_df(sorting=False, column_name='_'.join(['iter', str(max_iter)])))

rw_result_df_iters = pd.concat(rw_result_df_list, axis=1)

for column in rw_result_df_iters.columns:
    #plt.xlim(-2, 3)
    #plt.ylim(-2, 2)
    plt.scatter(x, y, c=rw_result_df_iters[column].rank(), s=0.05, cmap='Reds')
    plt.savefig('_'.join(['scatter_RW_iter', column]))
    plt.close()


# computing time simulation

RW = RandomWalk()

TSN = TimeSimulationNetwork(function=RW.run)
TSN.simulation_grid
TSN.time_it()

# GridTuneCV test
LOO = LOOCV(RandomWalk())
params = LOO.get_params()
params['gamma'] = [0.1, 0.5, 0.9]
params = {'max_iter': [1, 5, 10]}

cv_tune = GridTuneCV(base_cv=LOO, params_to_tune=params)
cv_tune.tune(G, seed_nodes)
cv_tune.get_results_df()

# LOO testing (PASS0
LOO = LOOCV(RandomWalk())
LOO.cv(G, seed_nodes)
LOO.set_params(params)
LOO.get_results_df()
LOO.get_params()




# RW testing (PASS)
RW = RandomWalk()
RW.run(G, seed_nodes)

RW.node_init_values
RW.node_final_values
RW.get_params()
params = RW.get_params()
params
RW.set_params(**RW.get_params())


RW.get_metrics(Metrics.rank)
RW.get_results_df()
RW.get_params()


# Testing (PASS)
mp.cpu_count()

# Test init_value_by_seed (PASS)
init_value_dict = init_value_by_seed(G=G, seed_nodes=seed_nodes)


# Playground
RW = RandomWalk()
RW1 = RW.copy()


RW.set_params(**{'gamma': 0.7, 'tol': 1e-08, 'max_iter': 100})
RW.get_params()
RW1.get_params()


class sumabc:
    def __init__(self, a):
        self.a = a

    def add_together(self, b):
        return self.a + b

    def mp_run(self):
        pool = mp.Pool(6)
        results = [pool.map(self.add_together,[i for i in range(5)])]
        pool.close()
        return results

test_class = sumabc(5)
test_class.mp_run()

pool = mp.Pool(6)
results = [pool.apply(sumabc, args=(i, 2, 3)) for i in range(6)]
pool.close()

bar = foo(3, 4)


for i,j in zip([4,5], [1,2,3]):
    print(i)
    print(j)


print(bar.value)
print(bar.square)

def give_a_warning():
    raise UserWarning("hahhahaha")
    pass


def expand_grid(params):
    """
    :param params: dict of range of params
    :return: df with all combinations, column names being names of params
    """
    rows = itertools.product(*params.values())
    return pd.DataFrame.from_records(rows, columns=params.keys())


x = {'a': 1, 'b': 2, 'c': 3}
np.mean([*x.values()])

y = itertools.product(*x.values())

def sumabc(a, b, c):
    return a+b+c
z = list()

for i in y:
    z.append(sumabc(**i))

expand_grid(x)


class addition:
    def __init__(self, a):
        self.a = a

    def adding_one(self):
        self.a += 1

a = 1


for key in x.keys():
    print(key)

pd.DataFrame(x)

y[list(x.keys())[0]] = list(x.values())[0]

gp_tools.methods.Metrics.rank(x, keys=['a', 'b'])

def xxx():
    cv_batch = namedtuple('cv_batch', ['train', 'test', 'name_batch'])
    a = cv_batch(1, 2, 'cv1')
    return a

df_test = pd.DataFrame.from_dict(x, orient='index')
df_test.columns = ['value']
df_test['value']
test_dict = [dict(row._asdict()) for row in df_test.itertuples()]

for i, j, k in [a]:
    print(i)
    print(j)
    print(k)

[i
 for i in range(5)]

sumabc(**x)



def random_graph_set(n, p, seed=None, directed=False):
    """
    The function to generate a set of random graphs

    :param n: list of number of nodes
    :param p: list of probability of edges
    :param seed: random state control
    :param directed: default False
    :return: list of graphs using combinations of n and p
    """

    random_graphs = []
    for nn in n:
        for pp in p:
            random_graphs.append(nx.erdos_renyi_graph(n=nn, p=pp))
    return random_graphs

G_set = random_graph_set(n = [100, 1000], p = [0.01])

lcc = nx.subgraph(G_set[0], nbunch=max(nx.connected_components(G_set[0]), key=len))

lcc.nodes

[len(graph) for graph in nx.connected_components(G_set[0])]
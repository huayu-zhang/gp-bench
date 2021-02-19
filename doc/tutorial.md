### Cloning the repo
Code tested only in linux. 

```shell script
git clone git@git.ecdf.ed.ac.uk:hzhang13/graph_gp.git
cd graph_gp
```

Or use http
```shell script
git clone https://git.ecdf.ed.ac.uk/hzhang13/graph_gp.git
cd graph_gp
```


<br/>

### Setup the environment with conda

[Install conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

 create the env
 ```shell script
conda env create -f environment.yml
```
<br/>

### Tutorial for a standard application
Our code use [*networkx*](https://networkx.github.io/) as the package to deal with graph structure.
<br/>

##### Running algorithms

```python
import sys
import os
import networkx as nx
import pandas as pd

sys.path.extend(os.getcwd())
from gp_tools.methods.randomwalk import RandomWalk
from gp_tools.methods.node2vec import Node2Vec

# import a protein-protein interaction graph
pp = nx.read_edgelist('./project_data/graphs/ppiall.edgelist')

# Seed genes for 
seed_genes = pd.read_csv('./project_data/data/seed_genes.tsv', sep='\t')
seed_nodes = list(seed_genes.seeds)

#############################
# Application of RandomWalk 


algorithm = RandomWalk()
print(algorithm.get_params()) # To see a dict of param you can set for this algorithm
# params = algorithm.get_params()
# algorithm.set_params(**params) # You can customize params here, you can also do this when you initiate the algorithm

algorithm.run(G=pp, seed_nodes=seed_nodes)

# RandomWalk algorithm give each node a score
print(algorithm.results) # dict of {gene id: score}
print(algorithm.get_results_df()) # datafrom with gene id and score

####################################
# Application of Node2Vec is similar
# But the node2vec is a multi-step algorithm (Please consult the node2vec citation for more details)
# Step 1: Create walks
# Step 2: Learn embeddings based on the walks
# Step 3: Calculate the similarity score to seed genes
# If you simply run the algorithm, all the steps will be run at once
# You can do it stepwise, which is helpful for cross-validation sometimes when you 
# want to tune parameters in later steps. You don't need to re-run earlier steps 

algorithm = Node2Vec()
algorithm.run(G=pp, seed_nodes=seed_nodes)

# Or do this stepwise with the same runtime
# Because once the walks done or embedding learnt, the algorithm just proceeds to next step

algorithm.simulate_walks(G=pp)
algorithm.learn_embeddings()
algorithm.run(G=pp, seed_nodes=seed_nodes)

# Getting results is the same for all algorithms
print(algorithm.results)
print(algorithm.get_results_df())
```
<br/>

##### Cross-validation
Only leave-one-out cross validation for now, but can added other CVs later easily

```python
import sys
import os
import networkx as nx
import pandas as pd

sys.path.extend(os.getcwd())
from gp_tools.methods.randomwalk import RandomWalk
from gp_tools.methods.cv_tuning import LOOCV

# import a protein-protein interaction graph
pp = nx.read_edgelist('./project_data/graphs/ppiall.edgelist')

# Seed genes
seed_genes = pd.read_csv('./project_data/data/seed_genes.tsv', sep='\t')
seed_nodes = list(seed_genes.seeds)

# Algorithm
algorithm = RandomWalk()

loocv = LOOCV(base_algorithm=algorithm) # n_jobs here controls threads used for paralel computing
loocv.cv(G=pp, seed_nodes=seed_nodes)

print(loocv.results)
print(loocv.get_results_df())
```
<br/>

##### Tuning hyperparameters
Hyperparameter tuning with cross-validation is implemented

```python
import sys
import os
import networkx as nx
import pandas as pd

sys.path.extend(os.getcwd())
from gp_tools.methods.randomwalk import RandomWalk
from gp_tools.methods.cv_tuning import LOOCV
from gp_tools.methods.cv_tuning import GridTuneCV

# import a protein-protein interaction graph
pp = nx.read_edgelist('./project_data/graphs/ppiall.edgelist')

# Seed genes
seed_genes = pd.read_csv('./project_data/data/seed_genes.tsv', sep='\t')
seed_nodes = list(seed_genes.seeds)

# Algorithm
algorithm = RandomWalk()
loocv = LOOCV(base_algorithm=algorithm) # n_jobs here controls threads used for paralel computing

params_to_tune = {
    'gamma': [0.1, 0.5, 0.9],
    'tol': 1e-08,
    'max_iter': [10, 100],
    'protein_identifier': 'ENS'
}

tune = GridTuneCV(base_cv=loocv, params_to_tune=params_to_tune)
tune.tune(G=pp, seed_nodes=seed_nodes)
print(tune.get_results_df())
```
<br/>

### Command line implementation
Command line implementation is provided for easier implementation on virtual machines

Help info
```shell script
python3 ./gp_tools_shell.py -h
```

Example implementation

First make the folder for the experiment and do the setup
The params can be changed then in .json file in ```./<experiment_dir>/params```

```shell script
mkdir example_command_line

python3 ./gp_tools_shell.py ./example_command_line/ RandomWalk --setup True

cp ./project_data/data/seed_genes.tsv ./example_command_line/data/seed_genes.tsv
cp ./project_data/graphs/ppiall.edgelist ./example_command_line/graphs/ppiall.edgelist

# Run with default settings
python3 ./gp_tools_shell.py ./example_command_line/ RandomWalk --name example
```

The output you should get is 

```text
Loading required files
Graph loaded: ./example_command_line/graphs/ppiall.edgelist
Number of nodes: 18718
Number of edges: 183457
Seed genes loaded: ./example_command_line/data/seed_genes.tsv
Seed nodes are:
['ENSG00000074181', 'ENSG00000187498', 'ENSG00000213689', 'ENSG00000166033', 'ENSG00000134871', 'ENSG00000093072', 'ENSG00000054598', 'ENSG00000164093', 'ENSG00000064601', 'ENSG00000130309']
Method parameters loaded ./example_command_line/params/parameters.json
Parameters are:
{'gamma': 0.5, 'tol': 1e-08, 'max_iter': 100, 'protein_identifier': 'ENS'}
Experiment parameter file saved: ./example_command_line/params/example_Wed_Sep_16_20_16_12_2020.json
Method loaded: Random_Walk
cv_method loaded: LOOCV
Starting cross-validation
Finshed cv batch cv_ENSG00000074181 (1/10)
Finshed cv batch cv_ENSG00000187498 (2/10)
Finshed cv batch cv_ENSG00000213689 (3/10)
Finshed cv batch cv_ENSG00000166033 (4/10)
Finshed cv batch cv_ENSG00000134871 (5/10)
Finshed cv batch cv_ENSG00000093072 (6/10)
Finshed cv batch cv_ENSG00000054598 (7/10)
Finshed cv batch cv_ENSG00000164093 (8/10)
Finshed cv batch cv_ENSG00000064601 (9/10)
Finshed cv batch cv_ENSG00000130309 (10/10)
Finished cross_validation
Cross-validation results saved at: ./example_command_line/results/example_Wed_Sep_16_20_16_12_2020.csv
Full results saved at: ./example_command_line/results/example_Wed_Sep_16_20_16_12_2020_full_results.csv
Cross-validation object dumped at: ./example_command_line/obj_dump/example_Wed_Sep_16_20_16_12_2020.pydump
```
<br/>

### Structure of the repo

All source codes are in ```gptools/``` directory

```text
gp_tools/    
    db/ # here are the classes to extract info from various databases
        raw/ # here for the raw files downloaded from the websites
            disgenet/...
            gtrd/...
            hpo/...
            huri/...            
            mimminer/...
            reactome/...
            stringdb/...
        edgelist/ # here the cleaned edgelist from these websites
            disgenet/...
            gtrd/...
            hpo/...
            huri/...            
            mimminer/...
            reactome/...
            stringdb/...
        __init__.py
        disgenet.py
        gtrd.py
        hpo.py
        huri.py
        mimminer.py
        reactome.py
        stringdb.py
    methods/ # here are the classes for algorithms and tuning of algorithms
        __init__.py
        cv_tuning.py
        diamond.py
        node2vec.py
        randomwalk.py
        genepanda.py
        idlp.py
        metrics.py
        utils.py
    __init__.py
    IO.py #
```


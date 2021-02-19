import argparse
import networkx as nx
import pandas as pd
import os
import importlib
import warnings
import json
import time
import pickle


# Get input edgelist, seed nodes, algorithm, results folder, dumping folder
def parse_args():
    """
    Parsing args for the algorithm run:
        graph_path: path of edgelist file
        seed_nodes: path of seed_nodes.tsv file
        algorithm: name of algorithm class
        result_path: path of results
        dumping_path: path for dump serialized object
    """
    parser = argparse.ArgumentParser(description="Command line implement for gene prioritization")
    parser.add_argument('ex_dir',
                        help='The directory to run the experiment')
    parser.add_argument('method',
                        help='prioritization method')
    parser.add_argument('--setup', default=False, type=bool,
                        help='True or False: running setup mode to create directories '
                             'and parameter list for default settings')
    parser.add_argument('-N', '--name', default='experiment',
                        help='name used by output')
    parser.add_argument('-G', '--graph', default='graphs/',
                        help='path of graph in format of .edgelist file\nOr the directory containing the file')
    parser.add_argument('-S', '--seed', default='data/',
                        help='path of seed list .tsv file\nOr the directory containing the file')
    parser.add_argument('-P', '--params', default='params/',
                        help='path of parameters .json file\nOr the directory containing the file')
    parser.add_argument('--cv_method', default='LOOCV',
                        help='cross-validation method for model internal evaluation\nIf cv not needed, set to None')
    parser.add_argument('--cv_n_jobs', default=1, type=int,
                        help='number of cpu cores used in cross-validation')
    parser.add_argument('--result_path', default='results/',
                        help='result path; default: results/')
    parser.add_argument('--dumping_path', default='obj_dump/',
                        help='dumping path; default: obj_dump/')

    return parser.parse_args()


def setup():
    args = parse_args()
    dir_list = [args.seed, args.graph, args.params, args.result_path, args.dumping_path]

    if not os.path.exists(args.ex_dir):
        os.mkdir(args.ex_dir)
        print('Experiment folder created: %s' % args.ex_dir)
    else:
        print('Experiment folder exists: %s' % args.ex_dir)

    # Create the folders in experiment directory: data/ graphs/ params/ results/ obj_dump/
    # if not os.path.exists(os.path.join(args.ex_dir, dir_list[0])):
    #     os.system('cp -r project_data/data %s' % args.ex_dir)
    #     print('seed file copied from ./project_data/data to %s' % (os.path.join(args.ex_dir, dir_list[0])))
    # else:
    #     print('Folder exists: %s' % os.path.join(args.ex_dir, dir_list[0]))
    #
    # if not os.path.exists(os.path.join(args.ex_dir, dir_list[1])):
    #     os.system('cp -r project_data/graphs %s' % args.ex_dir)
    #     print('graphs copied from ./project_data/data to %s' % os.path.join(args.ex_dir, dir_list[1]))
    # else:
    #     print('Folder exists: %s' % os.path.join(args.ex_dir, dir_list[1]))

    for directory in dir_list:
        if not os.path.exists(os.path.join(args.ex_dir, directory)):
            os.mkdir(os.path.join(args.ex_dir, directory))
            print('Folder created: %s' % os.path.join(args.ex_dir, directory))
        else:
            print('Folder exists: %s' % os.path.join(args.ex_dir, directory))

    # Get the default model params
    module = importlib.import_module('gp_tools.methods.%s' % args.method.lower())
    method_class = getattr(module, args.method)
    method = method_class()

    with open(os.path.join(args.ex_dir, 'params', 'parameters.json'), 'w') as f:
        f.write(json.dumps(method.get_params()))
    print('Parameter file created: %s' % os.path.join(args.ex_dir, 'params', 'parameters.json'))

    for edgelist in os.listdir(os.path.join(args.ex_dir, dir_list[1])):
        with open(os.path.join(args.ex_dir, '_'.join([args.method, edgelist,'.sh'])), 'w') as f:
            f.write('python gp_tools_shell.py %s %s --graph %s'
                    % (
                       args.ex_dir,
                       args.method,
                       os.path.join(args.ex_dir, dir_list[1], edgelist)
                    ))


def main():
    args = parse_args()

    if args.setup:
        setup()
    else:
        # find graph file
        if '.edgelist' not in args.graph:
            graph_files = [os.path.join(args.ex_dir, args.graph, file)
                           for file in os.listdir(os.path.join(args.ex_dir, args.graph)) if '.edgelist' in file]
            if len(graph_files) == 0:
                raise IOError('Graph .edgelist file not exist in <ex_dir>/<graph>')
            elif len(graph_files) == 1:
                graph_file = graph_files[0]
            else:
                graph_file = graph_files[0]
                warnings.warn('Multiple graph given: choose %s' % graph_files[0])
        else:
            graph_file = args.graph

        # file seed file
        if '.tsv' not in args.seed:
            seed_files = [os.path.join(args.ex_dir, args.seed, file)
                          for file in os.listdir(os.path.join(args.ex_dir, args.seed)) if '.tsv' in file]
            if len(seed_files) == 0:
                raise IOError('Seed .tsv file not exist in <ex_dir>/<data>')
            else:
                seed_file = seed_files[0]
        else:
            seed_file = args.seed

        # file param file
        if '.json' not in args.params:
            params_files = [os.path.join(args.ex_dir, args.params, file)
                            for file in os.listdir(os.path.join(args.ex_dir, args.params)) if '.json' in file]
            if len(params_files) == 0:
                params_file = None
                print('No method parameters given, default will be used')
            else:
                params_file = params_files[0]
        else:
            params_file = args.params

        # Tag time stamp in experiment name and create paths for result and dumping files
        args.name = '_'.join([args.name, time.asctime().replace(' ', '_').replace(':', '_')])
        result_file = os.path.join(args.ex_dir, args.result_path, args.name + '.csv')
        full_result_file = os.path.join(args.ex_dir, args.result_path, args.name + '_full_results' + '.csv')
        dumping_file = os.path.join(args.ex_dir, args.dumping_path, args.name + '.pydump')
        params_save_file = os.path.join(args.ex_dir, args.params, args.name + '.json')

        # input graph, seed, params
        print('Loading required files')
        G = nx.read_edgelist(graph_file)
        print('Graph loaded: %s' % graph_file, end='\n')
        print('Number of nodes: %s' % G.number_of_nodes(), end='\n')
        print('Number of edges: %s' % G.number_of_edges(), end='\n')

        seed_genes = pd.read_csv(seed_file, sep='\t')
        seed_nodes = list(seed_genes.seeds)
        print('Seed genes loaded: %s' % seed_file, end='\n')
        print('Seed nodes are:', end='\n')
        print(seed_nodes, end='\n')

        # load algorithm

        method_module = importlib.import_module('gp_tools.methods.%s' % args.method.lower())
        try:
            method_class = getattr(method_module, args.method)
        except AttributeError:
            raise AttributeError('method %s not exist' % args.method)
        algorithm = method_class()

        if args.params is not None:
            with open(params_file, 'r') as f:
                params = json.loads(f.readline())
            print('Method parameters loaded %s' % params_file, end='\n')
            print('Parameters are:', end='\n')
            print(params, end='\n')
            algorithm.set_params(**params)

            with open(params_save_file, 'w') as f:
                f.write(json.dumps(algorithm.get_params()))
            print('Experiment parameter file saved: %s' % params_save_file)

        print('Method loaded: %s' % algorithm.algorithm_name, end='\n')

        # Execution

        if args.cv_method != 'None':
            cv_module = importlib.import_module('gp_tools.methods.cv_tuning')
            try:
                cv_class = getattr(cv_module, args.cv_method)
            except AttributeError:
                raise AttributeError('cv_method %s not exist' % args.cv_method)
            cv = cv_class(base_algorithm=algorithm, n_jobs=args.cv_n_jobs)
            print('cv_method loaded: %s' % args.cv_method, end='\n')

            print('Starting cross-validation', end='\n')
            cv.cv(G, seed_nodes)
            print('Finished cross_validation', end='\n')

            cv.get_results_df().to_csv(result_file)
            print('Cross-validation results saved at: %s' % result_file, end='\n')
            cv.get_full_results_df().to_csv(full_result_file)
            print('Full results saved at: %s' % full_result_file, end='\n')

            with open(dumping_file, 'wb') as f:
                pickle.dump(cv, f)
            print('Cross-validation object dumped at: %s' % dumping_file, end='\n')

        else:
            print('Starting algorithm', end='\n')
            algorithm.run(G, seed_nodes)
            print('Finished algorithm', end='\n')

            algorithm.get_results_df().to_csv()
            print('Method results saved at: %s' % result_file, end='\n')

            with open(dumping_file, 'wb') as f:
                pickle.dump(algorithm, f)
            print('Method object dumped at: %s' % dumping_file, end='\n')


if __name__ == '__main__':
    main()

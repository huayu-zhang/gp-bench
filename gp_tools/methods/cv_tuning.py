from gp_tools.methods.metrics import ModelMetrics
from gp_tools.methods.utils import param_to_name
from gp_tools.IO import pickle_dump_to_file
import os
import multiprocessing as mp
import pandas as pd
import itertools
import numpy as np
import copy


def expand_grid(params):
    """
    :param params: dict of range of paramsrw_value_df
    :return: df with all combinations, column names being names of params
    """
    for key in params.keys():
        if not isinstance(params[key], list):
            params[key] = [params[key]]
    iter_df = pd.DataFrame.from_records(itertools.product(*params.values()), columns=params.keys())
    return iter_df


def combine_params(new, existing):
    if not set(new.keys()).issubset(set(existing.keys())):
        raise UserWarning("Not a tunable parameter in the algorithm: %s" % (
            set(new.keys()).difference(set(existing.keys()))
        ))

    for key in existing.keys():
        if key not in set(new.keys()):
            new[key] = existing[key]

    return new


def loocv_experiment(graph,
                     graph_name,
                     seed_nodes,
                     algorithm,
                     dump_path,
                     result_path,
                     param_range,
                     n_jobs=1,
                     cv_func=None
                     ):

    if not os.path.exists(os.path.join(dump_path, graph_name)):
        os.mkdir(os.path.join(dump_path, graph_name))

    algorithm_name = algorithm.algorithm_name.replace('_', '')

    if not os.path.exists(os.path.join(dump_path, graph_name, algorithm_name)):
        os.mkdir(os.path.join(dump_path, graph_name, algorithm_name))

    current_dump_dir = os.path.join(dump_path, graph_name, algorithm_name)

    params_to_tune = expand_grid(
        combine_params(new=param_range, existing=algorithm.get_params())
    )

    tune_results = []

    for param in params_to_tune.to_dict('records'):

        # Get name
        name_current_iter = param_to_name(param)
        print('Current params: %s' % name_current_iter)

        if cv_func is None:
            # Run LOOCV
            base_algorithm = copy.deepcopy(algorithm)
            base_algorithm.set_params(**param)
            loocv = LOOCV(base_algorithm=base_algorithm, n_jobs=n_jobs)
            loocv.cv(graph, seed_nodes)
        else:
            loocv = cv_func(graph,
                            seed_nodes,
                            param,
                            algorithm,
                            graph_name,
                            n_jobs
                            )

        # Append results, dump LOOCV
        tune_results.append(loocv.get_results_df(index_name=name_current_iter))
        pickle_dump_to_file(loocv, os.path.join(current_dump_dir, name_current_iter + '.pydump'))
        print('LOOCV file dumped at: %s' % os.path.join(current_dump_dir, name_current_iter + '.pydump'))

    results_df = pd.concat(tune_results)
    results_df.index = range(len(results_df.index))

    params_results = pd.concat([params_to_tune, results_df], axis=1)
    params_results['mean_rank'] = results_df.apply(np.mean, axis=1)
    params_results['median_rank'] = results_df.apply(np.median, axis=1)
    params_results.to_csv(os.path.join(result_path, '%s_%s.csv' % (graph_name, algorithm_name)))

    return params_results


def n2v_cv_func(G, seed_nodes, param, algorithm, graph_name, n_jobs):
    dump_name = 'graph-%s_p-%2.1f_q-%2.1f_number_walks-%s_walk_length-%s_dim-%s' % \
                (graph_name, param['p'], param['q'], param['number_walks'], param['walk_length'], param['dimensions'])

    base_algorithm = copy.deepcopy(algorithm)
    base_algorithm.set_params(**param)
    base_algorithm.learn_embeddings(dumped_embedding=os.path.join('./experiment_20200108/vec_dump', dump_name))

    print('Embeddings loaded: %s' % dump_name)

    loocv = LOOCV(base_algorithm=base_algorithm, n_jobs=n_jobs)
    loocv.cv(G, seed_nodes)

    return loocv


# Cross validation utilities

class LOOCV:

    @staticmethod
    def LOO_seed_spliter(seed_nodes):
        """
        Split the list of seed nodes into all-but-one training set and the left out test set
        Results in tuple(trainlist, testlist, name_batch)
        :param seed_nodes: list of seed nodes
        :return: [tuple1, tuple2, .....]
        """
        seed_cv_list = []
        total_batch = len(seed_nodes)

        for index, node in enumerate(seed_nodes):
            temp_nodes = seed_nodes.copy()
            temp_nodes.remove(node)
            seed_cv_list.append(
                (
                    temp_nodes,
                    [node],
                    '_'.join(['cv', str(node)]),
                    '(%s/%s)' % (index + 1, total_batch)
                 )
            )

        return seed_cv_list

    def __init__(self,
                 base_algorithm,
                 params=None,
                 n_jobs=1,
                 metrics='rank'):
        self.base_algorithm = base_algorithm
        self.n_jobs = n_jobs

        if params is None:
            self.params = self.base_algorithm.get_params()
        else:
            self.params = params
            self.base_algorithm.set_params(**self.params)

        if isinstance(metrics, str):
            self.metrics_name = metrics
            try:
                self.metrics_function = getattr(ModelMetrics, metrics)
            except AttributeError:
                raise AttributeError('metrics %s is not predefined' % self.metrics_name)
        else:
            self.metrics_name = 'user_defined'
            if callable(metrics):
                self.metrics_function = metrics
            else:
                raise TypeError('metrics should be either defined method in class Metrics or a function')

        self.seed_nodes = None
        self.cv_node_list = None
        self.results = None
        self.full_results = None

    def get_cv_params(self):
        params = {
            'base_algorithm': self.base_algorithm,
            'params': self.params,
            'n_jobs': self.n_jobs,
            'metrics': self.metrics_name
        }
        return params

    def copy(self):
        return LOOCV(**self.get_cv_params())

    def set_params(self, new_params):
        self.params = new_params
        self.base_algorithm.set_params(**self.params)

    def get_params(self):
        return self.params

    def get_cv_node_list(self):
        return self.cv_node_list

    def _cv_core(self, G, train, test, name_batch, index_batch):
        algorithm = copy.deepcopy(self.base_algorithm)

        if G.has_edge(test[0], '$$$target_node'):
            G_temp = G.copy()
            G_temp.remove_edge(test[0], '$$$target_node')
            algorithm.run(G_temp, train)
            print('Edge %s - %s removed to avoid data leaking' % (test[0], '$$$target_node'), end='\n')
        else:
            algorithm.run(G, train)
        result_metrics = self.metrics_function(algorithm.results, test)
        result_df = algorithm.get_results_df(sorting=False, column_name='_'.join([self.metrics_name, name_batch]))

        print('Finshed cv batch %s %s' % (name_batch, index_batch))
        return result_metrics, result_df, name_batch

    def cv(self, G, seed_nodes):

        self.seed_nodes = seed_nodes
        self.cv_node_list = self.LOO_seed_spliter(seed_nodes)

        self.results = dict()
        self.full_results = list()

        if self.n_jobs <= 1:
            results = []
            for train, test, name_batch, index_batch in self.cv_node_list:
                results.append(self._cv_core(G, train, test, name_batch, index_batch))
        else:
            if self.n_jobs > len(os.sched_getaffinity(0)):
                self.n_jobs = len(os.sched_getaffinity(0))
            map_list = [(G, train, test, name_batch, index_batch)
                        for train, test, name_batch, index_batch in self.cv_node_list]
            pool = mp.Pool(self.n_jobs)
            results = pool.starmap(self._cv_core, map_list)
            pool.close()

        for result_metrics, result_df, name_batch in results:
            self.results['_'.join([name_batch, self.metrics_name])] = list(result_metrics.values())[0]
            self.full_results.append(result_df)

    def get_results_df(self, index_name=None):
        df = pd.DataFrame.from_dict(self.results, orient='index')
        if index_name is not None:
            df.columns = [index_name]
        return df.transpose()

    def get_full_results_df(self):
        return pd.concat(self.full_results, axis=1)


# Model tuning utilities

class GridTuneCV:

    @staticmethod
    def expand_grid(params):
        """
        :param params: dict of range of paramsrw_value_df
        :return: df with all combinations, column names being names of params
        """
        for key in params.keys():
            if not isinstance(params[key], list):
                params[key] = [params[key]]
        iter_df = pd.DataFrame.from_records(itertools.product(*params.values()), columns=params.keys())
        return iter_df

    @staticmethod
    def combine_params(new, existing):

        if not set(new.keys()).issubset(set(existing.keys())):
            raise UserWarning("Not a tunable parameter in the algorithm: %s" % (
                set(new.keys()).difference(set(existing.keys()))
            ))

        for key in existing.keys():
            if key not in set(new.keys()):
                new[key] = existing[key]

        return new

    def __init__(self,
                 base_cv,
                 params_to_tune,
                 metrics='minus_rank_ratio',
                 n_jobs=1
                 ):
        self.base_cv = base_cv
        self.params_to_tune = self.combine_params(new=params_to_tune, existing=self.base_cv.get_params())
        self.tune_grid = self.expand_grid(self.params_to_tune)
        self._tune_dicts = self.tune_grid.to_dict('records')
        self.n_jobs = n_jobs
        self.results = None
        self.results_df = None
        self.metrics = None

        if isinstance(metrics, str):
            self.metrics_name = metrics
            try:
                self.metrics_function = getattr(ModelMetrics, metrics)
            except AttributeError:
                raise AttributeError('metrics %s is not predefined' % self.metrics_name)
        else:
            self.metrics_name = 'user_defined'
            if callable(metrics):
                self.metrics_function = metrics
            else:
                raise TypeError('metrics should be either defined method in class Metrics or a function')

    def set_params_to_tune(self, new_params_to_tune):
        self.params_to_tune = self.combine_params(new=new_params_to_tune, existing=self.base_cv.get_params())
        self.tune_grid = self.expand_grid(self.params_to_tune)
        self._tune_dicts = self.tune_grid.to_dict('records')

    def get_tune_grid(self):
        return self.tune_grid

    def _tune_core(self, G, seed_nodes, params):
        temp_cv = self.base_cv.copy()
        temp_cv.set_params(params)
        temp_cv.cv(G, seed_nodes)
        return temp_cv.results

    def tune(self, G, seed_nodes):
        if self.n_jobs <= 1:
            self.results = [self._tune_core(G, seed_nodes, params) for params in self._tune_dicts]
        else:
            if self.n_jobs > len(os.sched_getaffinity(0)):
                self.n_jobs = len(os.sched_getaffinity(0))
            pool = mp.Pool(self.n_jobs)
            self.results = [pool.apply(self._tune_core, args=(G, seed_nodes, params))
                            for params in self._tune_dicts]
            pool.close()

        self.results_df = pd.concat(
            [self.tune_grid, pd.DataFrame.from_records(self.results, index=range(len(self.results)))],
            axis=1
        )

        self.metrics = self.metrics_function(self.results)
        self.results_df[self.metrics_name] = self.metrics

    def get_results_df(self):
        return self.results_df

    def best_params(self):
        return self._tune_dicts[np.argmin(self.metrics)]


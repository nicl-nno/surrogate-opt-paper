import csv
import datetime
import os
import random
from functools import partial
from itertools import repeat
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from src.basic_evolution.errors import (
    error_rmse_all,
    error_mae_all,
    error_mae_peak,
    error_rmse_peak
)
from src.basic_evolution.evo_operators import (
    calculate_objectives_interp,
    crossover,
    mutation,
    initial_pop_lhs
)
from src.basic_evolution.model import (
    CSVGridFile,
    FidelityFakeModel
)
from src.basic_evolution.model import SWANParams

from src.evolution.operators import default_operators
from src.evolution.spea2.dynamic import DynamicSPEA2
from src.evolution.spea2.spea2 import SPEA2
from src.multifidelity_evolution.fidelity_handler import FidelityHandler

from src.utils.files import (
    wave_watch_results
)
from src.utils.vis import (
    plot_results,
    plot_population_movement
)

ALL_STATIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

np.random.seed(43)
random.seed(43)


def get_rmse_for_all_stations(forecasts, observations):
    time = np.linspace(1, 253, num=len(forecasts[0].hsig_series))

    obs_series = []
    for obs in observations:
        obs_series.append(obs.time_series(from_date="20140814.120000", to_date="20140915.000000")[:len(time)])

    results_for_stations = np.zeros(len(forecasts))
    for idx in range(0, len(forecasts)):
        results_for_stations[idx] = np.sqrt(
            np.mean((np.array(forecasts[idx].hsig_series) - np.array(obs_series[idx])) ** 2))
    return results_for_stations


def model_all_stations(forecasts_range=(0, 1)):
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')
    ww3_obs = \
        [obs.time_series() for obs in
         wave_watch_results(path_to_results='../../samples/ww-res/', stations=ALL_STATIONS)]

    model = FidelityFakeModel(grid_file=grid, observations=ww3_obs, stations_to_out=ALL_STATIONS, error=error_rmse_all,
                              forecasts_path='../../../2fidelity/*', forecasts_range=forecasts_range,
                              is_surrogate=False)

    return model


def default_params_forecasts(model):
    '''
    Our baseline:  forecasts with default SWAN params
    '''

    closest_params = model.closest_params(params=SWANParams(drf=1.0,
                                                            cfw=0.015,
                                                            stpm=0.00302))
    default_params = SWANParams(drf=closest_params[0], cfw=closest_params[1], stpm=closest_params[2])
    drf_idx, cfw_idx, stpm_idx, fid_time_idx, fid_space_idx = model.params_idxs(default_params)
    forecasts = model.grid[drf_idx, cfw_idx, stpm_idx, fid_time_idx, fid_space_idx]

    return forecasts


def save_archive_history(history, file_name='history.csv'):
    test_model = model_all_stations()

    with open(file_name, 'w', newline='') as csvfile:
        fieldnames = ['idx', 'gen_idx'] + [f'err_{idx + 1}' for idx in range(len(ALL_STATIONS))] \
                     + ['drf', 'stpm', 'cfw']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for gen_idx, gen in enumerate(history):
            for ind_idx, ind in enumerate(gen):
                row_to_write = {}
                idx = gen_idx * len(gen) + ind_idx

                row_to_write['idx'] = idx
                row_to_write['gen_idx'] = gen_idx

                params = test_model.closest_params(ind.genotype)
                closest_params = SWANParams(drf=params[0], cfw=params[1], stpm=params[2],
                                            fidelity_time=params[3], fidelity_space=params[4])

                # TODO: bug here
                metrics = test_model.output(params=closest_params)
                for err_idx, err_value in enumerate(metrics):
                    row_to_write[f'err_{err_idx + 1}'] = err_value

                row_to_write['drf'] = ind.genotype.drf
                row_to_write['stpm'] = ind.genotype.stpm
                row_to_write['cfw'] = ind.genotype.cfw

                writer.writerow(row_to_write)


def run_genetic_opt(max_gens, pop_size, archive_size, crossover_rate, mutation_rate, mutation_value_rate, sur_points,
                    stations,
                    **kwargs):
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')
    ww3_obs = \
        [obs.time_series() for obs in wave_watch_results(path_to_results='../../samples/ww-res/', stations=stations)]

    train_model = FidelityFakeModel(grid_file=grid, observations=ww3_obs, stations_to_out=stations,
                                    error=error_rmse_all,
                                    forecasts_path='../../../2fidelity/*', is_surrogate=True, sur_points=sur_points,
                                    initial_fidelity=(180, 56))
    test_range = (0, 1)
    test_model = model_all_stations(forecasts_range=test_range)

    operators = default_operators()

    handler = FidelityHandler(surrogates=train_model.surrogates_by_stations, time_delta=30, space_delta=14,
                              point_for_retrain=3, gens_to_change_fidelity=10)

    history, archive_history, _ = DynamicSPEA2(
        params=SPEA2.Params(max_gens=max_gens, pop_size=30, archive_size=10,
                            crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                            mutation_value_rate=mutation_value_rate),
        objectives=partial(calculate_objectives_interp, train_model),
        evolutionary_operators=operators,
        fidelity_handler=handler).solution(verbose=False)

    exptime2 = str(datetime.datetime.now().time()).replace(":", "-")
    # save_archive_history(archive_history, f'rob-exp-bl-{exptime2}.csv')

    params = history.last().genotype

    return history.last()


stations_for_train_set = [1, 2, 3]


def experiment_run(param_for_run, add_id, path_to_results):
    iterations = 8
    run_by = 'rmse_all'

    file_name = f'bl-{run_by}-add3_{add_id}-runs.csv'
    with open(os.path.join(path_to_results, file_name), 'w', newline='') as csvfile:
        fieldnames = ['ID', 'IterId', 'SetId', 'drf', 'cfw', 'stpm',
                      'rmse_all', 'rmse_peak', 'mae_all', 'mae_peak']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

    models_to_tests = init_models_to_tests()

    cpu_count = 1

    all_packed_params = []

    for st_set_id, stations_to_train, params in zip(list(range(iterations)), repeat(stations_for_train_set),
                                                    repeat(param_for_run)):
        all_packed_params.append([st_set_id, stations_to_train, params])

    results = []
    with Pool(processes=cpu_count) as p:
        with tqdm(total=iterations) as progress_bar:
            for _, out in tqdm(enumerate(p.imap(opt_run, all_packed_params))):
                results.append(out)
                progress_bar.update()

    for out in results:
        run_id, best = out
        with open(os.path.join(path_to_results, file_name), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            row_to_write = {'ID': run_id, 'IterId': 0, 'SetId': 0,
                            'drf': best.genotype.drf,
                            'cfw': best.genotype.cfw,
                            'stpm': best.genotype.stpm}
            metrics = all_error_metrics(best.genotype, models_to_tests)
            for metric_name in metrics.keys():
                stations_metrics = metrics[metric_name]
                stations_to_write = {}
                for station_idx in range(len(stations_metrics)):
                    key = f'st{station_idx + 1}'
                    stations_to_write.update({key: stations_metrics[station_idx]})
                row_to_write.update({metric_name: stations_to_write})

            writer.writerow(row_to_write)


def opt_run(packed_args):
    st_set_id, stations_for_run, param_for_run = packed_args
    print(stations_for_run)
    archive_size = round(param_for_run['archive_size_rate'] * param_for_run['pop_size'])
    mutation_value_rate = [param_for_run['mutation_p1'], param_for_run['mutation_p2'],
                           param_for_run['mutation_p3']]
    best = run_genetic_opt(max_gens=param_for_run['max_gens'],
                           pop_size=param_for_run['pop_size'],
                           archive_size=archive_size,
                           crossover_rate=param_for_run['crossover_rate'],
                           mutation_rate=param_for_run['mutation_rate'],
                           mutation_value_rate=mutation_value_rate,
                           sur_points=param_for_run['sur_points'],
                           stations=stations_for_run,
                           save_figures=False)

    return st_set_id, best


def init_models_to_tests():
    metrics = {'rmse_all': error_rmse_all,
               'rmse_peak': error_rmse_peak,
               'mae_all': error_mae_all,
               'mae_peak': error_mae_peak}

    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')
    ww3_obs = \
        [obs.time_series() for obs in
         wave_watch_results(path_to_results='../../samples/ww-res/', stations=ALL_STATIONS)]

    models = {}
    for metric_name in metrics.keys():
        model = FidelityFakeModel(grid_file=grid, observations=ww3_obs, stations_to_out=ALL_STATIONS,
                                  error=metrics[metric_name],
                                  forecasts_path='../../../2fidelity/*')
        models[metric_name] = model

    return models


def all_error_metrics(params, models_to_tests):
    metrics = {'rmse_all': error_rmse_all,
               'rmse_peak': error_rmse_peak,
               'mae_all': error_mae_all,
               'mae_peak': error_mae_peak}

    out = {}

    for metric_name in metrics.keys():
        model = models_to_tests[metric_name]

        out[metric_name] = model.output(params=params)

    return out


def reference_metrics():
    return all_error_metrics(params=SWANParams(drf=1.0, cfw=0.015, stpm=0.00302),
                             models_to_tests=init_models_to_tests())


def multiple_runs():
    exptime = str(datetime.datetime.now().time()).replace(":", "-")
    path_to_results = f'../../multiple_runs_{exptime}'
    os.mkdir(path_to_results)
    ind = 0
    for sur_points in range(50, 950, 100):
        # sur_points=5
        print(sur_points)
        objective_manual = {'a': 0, 'archive_size_rate': 0.25, 'crossover_rate': 0.7,
                            'max_gens': 30, 'mutation_p1': 0.1, 'mutation_p2': 0.01,
                            'mutation_p3': 0.001, 'mutation_rate': 0.7, 'pop_size': 30, 'sur_points': sur_points}

        experiment_run(objective_manual, ind, "C:\\Users\\Nikolay\\add-sur-dyn2")
        ind = ind + 1


if __name__ == '__main__':
    objective_manual = {'a': 0, 'archive_size_rate': 0.25, 'crossover_rate': 0.7,
                        'max_gens': 100, 'mutation_p1': 0.1, 'mutation_p2': 0.01,
                        'mutation_p3': 0.001, 'mutation_rate': 0.7, 'pop_size': 30, 'sur_points': 100}

    # opt_run([1,[1],objective_manual])

    multiple_runs()

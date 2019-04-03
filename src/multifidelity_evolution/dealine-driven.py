import numpy as np
from scipy.optimize import minimize, differential_evolution
from src.multifidelity_evolution.kriging_evolution import optimize_test

from functools import partial
from src.basic_evolution.errors import (
    error_rmse_all,
    error_mae_all,
    error_mae_peak,
    error_rmse_peak
)

from src.basic_evolution.evo_operators import (
    calculate_objectives_interp
)
from src.basic_evolution.model import (
    CSVGridFile,
    FidelityFakeModel,
    SWANPerfModel
)
from src.evolution.operators import default_operators
from src.evolution.spea2.dynamic import DynamicSPEA2, DynamicSPEA2PerfModel
from src.evolution.spea2.spea2 import SPEA2
from src.multifidelity_evolution.fidelity_handler import FidelityHandler
from src.utils.files import (
    wave_watch_results
)

import os
import csv

score_history = []
best_score_history = []

run_id = 0

ALL_STATIONS = [1, 2, 3]


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


models_to_tests = init_models_to_tests()


def run_evolution(sur_points, time_delta, space_delta, point_for_retrain, gens_to_change_fidelity, max_gens, pop_size,
                  archive_size, iter_id, deadline, points_by_fid):
    train_stations = [1, 2, 3]

    initial_fidelity = (180, 56)

    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')

    ww3_obs = \
        [obs.time_series() for obs in
         wave_watch_results(path_to_results='../../samples/ww-res/', stations=train_stations)]

    train_model = FidelityFakeModel(grid_file=grid, observations=ww3_obs,
                                    stations_to_out=train_stations, error=error_rmse_all,
                                    forecasts_path='../../../2fidelity/*', forecasts_range=(0, 1),
                                    sur_points=sur_points,
                                    is_surrogate=True, initial_fidelity=initial_fidelity)

    operators = default_operators()

    handler = FidelityHandler(surrogates=train_model.surrogates_by_stations, time_delta=time_delta,
                              space_delta=space_delta,
                              point_for_retrain=point_for_retrain, gens_to_change_fidelity=gens_to_change_fidelity)

    dyn_params = SPEA2.Params(max_gens=max_gens, pop_size=pop_size, archive_size=archive_size,
                              crossover_rate=0.7, mutation_rate=0.7,
                              mutation_value_rate=[0.1, 0.01, 0.001])

    history, _, points_by_fid_new = DynamicSPEA2(
        params=dyn_params,
        objectives=partial(calculate_objectives_interp, train_model),
        evolutionary_operators=operators,
        fidelity_handler=handler,
        points_by_fidelity=points_by_fid).solution(verbose=True)

    best = history.last()

    fieldnames = ['ID', 'IterId', 'SetId', 'drf', 'cfw', 'stpm',
                  'rmse_all', 'rmse_peak', 'mae_all', 'mae_peak', 'deadline']

    with open(os.path.join("C:\\metaopt2", "res-dd.csv"), 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        row_to_write = {'ID': run_id, 'IterId': iter_id, 'SetId': 0,
                        'drf': best.genotype.drf,
                        'cfw': best.genotype.cfw,
                        'stpm': best.genotype.stpm,
                        'deadline': deadline}
        metrics = all_error_metrics(best.genotype, models_to_tests)
        for metric_name in metrics.keys():
            stations_metrics = metrics[metric_name]
            stations_to_write = {}
            for station_idx in range(len(stations_metrics)):
                key = f'st{station_idx + 1}'
                stations_to_write.update({key: stations_metrics[station_idx]})
            row_to_write.update({metric_name: stations_to_write})

        writer.writerow(row_to_write)

    return history.last().error_value, points_by_fid_new


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


############################################

iter_id = 0
fieldnames = ['ID', 'IterId', 'SetId', 'drf', 'cfw', 'stpm',
              'rmse_all', 'rmse_peak', 'mae_all', 'mae_peak', 'deadline']

with open(os.path.join("C:\\metaopt2", "res2-dd.csv"), 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()


for iterId in range(100):
    min_deadline = 100
    for deadline_modifier in range(min_deadline, 1000, 100):
        deadline_ratio = deadline_modifier / min_deadline

        deadline = deadline_ratio * 60 * 60

        wasted_time = 0

        pre_calculated = 0

        initial_surrogate_points = round(10 + (20 * deadline_ratio - 1))

        saved_for_next = 0

        run_cons_id = 0

        points_prev=[]
        while True:
            print(f'iter {iterId}')
            print(f'dd {deadline}')
            print(f'runid {run_cons_id}')
            initial_fidelity = (180, 56)
            pop_size = 10 + 5 * (deadline_ratio - 1)
            max_gens = 10 + 5 * (deadline_ratio - 1)

            time_delta = 30
            space_delta = 14
            pop_size = int(round(pop_size))
            archive_size = round(pop_size / 2)
            point_for_retrain = round(archive_size / 4)
            gens_to_change_fidelity = round(max_gens / 2)

            handler = FidelityHandler(surrogates=[], time_delta=time_delta, space_delta=space_delta,
                                      point_for_retrain=point_for_retrain,
                                      gens_to_change_fidelity=gens_to_change_fidelity)

            dyn_params = SPEA2.Params(max_gens=max_gens, pop_size=pop_size, archive_size=archive_size,
                                      crossover_rate=0.7, mutation_rate=0.7,
                                      mutation_value_rate=[0.1, 0.01, 0.001])

            ex_time = DynamicSPEA2PerfModel.get_execution_time(initial_surrogate_points, initial_fidelity, dyn_params,
                                                               handler)

            evo_res, points = run_evolution(initial_surrogate_points, time_delta, space_delta, point_for_retrain,
                                            gens_to_change_fidelity, max_gens, pop_size,
                                            archive_size, iterId, deadline,points_prev)

            points_prev = points

            wasted_time = wasted_time + ex_time - ex_time * 0.9

            initial_surrogate_points += 50

            if wasted_time >= deadline:
                break
            run_cons_id = run_cons_id + 1

import numpy as np
from scipy.optimize import minimize, differential_evolution
from src.multifidelity_evolution.kriging_evolution import optimize_test

from functools import partial

from src.basic_evolution.errors import (
    error_rmse_all
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

deadline = 20 * 60 * 60

# SWANPerfModel.get_execution_time((14,60))
run_id = 0


def run_evolution(sur_points, time_delta, space_delta, point_for_retrain, gens_to_change_fidelity,max_gens,pop_size,archive_size):
    train_stations = [1, 2, 3]

    initial_fidelity = (180, 56)
    #sur_points = 5

    dyn_params = SPEA2.Params(max_gens=100, pop_size=10, archive_size=5,
                              crossover_rate=0.7, mutation_rate=0.7,
                              mutation_value_rate=[0.1, 0.01, 0.001])

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

    handler = FidelityHandler(surrogates=train_model.surrogates_by_stations, time_delta=15, space_delta=7,
                              point_for_retrain=3, gens_to_change_fidelity=10)

    dyn_params = SPEA2.Params(max_gens=max_gens, pop_size=pop_size, archive_size=archive_size,
                              crossover_rate=0.7, mutation_rate=0.7,
                              mutation_value_rate=[0.1, 0.01, 0.001])

    # ex_time = DynamicSPEA2PerfModel.get_execution_time(sur_points, initial_fidelity, dyn_params, handler)

    # print(ex_time)

    history, _ = DynamicSPEA2(
        params=dyn_params,
        objectives=partial(calculate_objectives_interp, train_model),
        evolutionary_operators=operators,
        fidelity_handler=handler).solution(verbose=True)

    fieldnames = ['ID', 'deadline', 'sur_points', 'gens_to_change_fidelity', 'max_gens', 'pop_size', 'error']

    with open(os.path.join("C:\\Users\\Nikolay\\metaopt1", "res1.csv"), 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        row_to_write = {'ID': 0,
                        'deadline': deadline,
                        'sur_points': sur_points,
                        'gens_to_change_fidelity': gens_to_change_fidelity,
                        'max_gens': max_gens,
                        'pop_size': pop_size,
                        'error': history.last().error_value}

        writer.writerow(row_to_write)

    return history.last().error_value


def objective(args):
    sur_points, gens_to_change_fidelity, max_gens, pop_size = args

    sur_points = int(round(sur_points))

    time_delta = 30
    space_delta = 14
    pop_size = int(round(pop_size))
    archive_size = round(pop_size / 2)
    point_for_retrain = round(archive_size / 2)
    # int(round(point_for_retrain))
    gens_to_change_fidelity = int(round(gens_to_change_fidelity))

    handler = FidelityHandler(surrogates=[], time_delta=15, space_delta=7,
                              point_for_retrain=3, gens_to_change_fidelity=10)

    initial_fidelity = (180, 56)

    dyn_params = SPEA2.Params(max_gens=100, pop_size=10, archive_size=5,
                              crossover_rate=0.7, mutation_rate=0.7,
                              mutation_value_rate=[0.1, 0.01, 0.001])

    ex_time = DynamicSPEA2PerfModel.get_execution_time(sur_points, initial_fidelity, dyn_params, handler)
    if (ex_time >= deadline):
        print(ex_time)
        return 99999

    print("OBJ", sur_points, point_for_retrain, gens_to_change_fidelity)

    score = run_evolution(sur_points, time_delta, space_delta, point_for_retrain, gens_to_change_fidelity,max_gens,pop_size,archive_size)

    if (len(best_score_history) == 0 or score < best_score_history[len(best_score_history) - 1][
        0]):
        best_score = score
        best_score_history.append([best_score, ])
    score_history.append([score, ])

    print("SCORE FOUND")
    print(score)
    return score

    # optimize

    fieldnames = ['ID', 'deadline', 'sur_points', 'gens_to_change_fidelity', 'max_gens', 'pop_size', 'error']

    # run_id+=1

    writer = csv.DictWriter(os.path.join("C:\\Users\\Nikolay\\metaopt1", "res1.csv"), 'a', newline='',
                            fieldnames=fieldnames)
    writer.writeheader()


sur_points = (10, 200)
gens_to_change_fidelity = (3, 9)
max_gens = (10, 30)
pop_size = (20, 100)

deadline = 20 * 60 * 60
bnds = (sur_points, gens_to_change_fidelity, max_gens, pop_size)

for i in range (20):

    solution = differential_evolution(objective, bounds=bnds, maxiter=200)
    x = solution.x

    # show final objective
    print('Final SSE Objective: ' + str(objective(x)))

    # print solution
    print('Solution')
    print('x1 = ' + str(x[0]))
    print('x2 = ' + str(x[1]))
    print('x3 = ' + str(x[2]))
    print('x4 = ' + str(x[2]))

    deadline = deadline * 2

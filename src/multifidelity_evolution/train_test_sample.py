import random
from functools import partial

import numpy as np

from src.basic_evolution.errors import (
    error_rmse_all
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
from src.evolution.spea2 import SPEA2
from src.utils.files import (
    wave_watch_results
)
from src.utils.vis import (
    plot_results,
    plot_population_movement
)

ALL_STATIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

np.random.seed(42)
random.seed(42)


def model_all_stations(forecasts_range=(0, 1)):
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')
    ww3_obs = \
        [obs.time_series() for obs in
         wave_watch_results(path_to_results='../../samples/ww-res/', stations=ALL_STATIONS)]

    model = FidelityFakeModel(grid_file=grid, observations=ww3_obs, stations_to_out=ALL_STATIONS, error=error_rmse_all,
                              forecasts_path='../../../2fidelity/*', forecasts_range=forecasts_range)

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


def optimize_test(train_stations, max_gens, pop_size, archive_size, crossover_rate, mutation_rate,
                  mutation_value_rate, plot_figures=True):
    train_range = (0, 0.5)
    test_range = (0.5, 1)

    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')

    ww3_obs = \
        [obs.time_series() for obs in
         wave_watch_results(path_to_results='../../samples/ww-res/', stations=train_stations)]

    error = error_rmse_all
    train_model = FidelityFakeModel(grid_file=grid, observations=ww3_obs, stations_to_out=train_stations, error=error,
                                    forecasts_path='../../../2fidelity/*', forecasts_range=train_range)

    history, archive_history = SPEA2(
        params=SPEA2.Params(max_gens, pop_size=pop_size, archive_size=archive_size,
                            crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                            mutation_value_rate=mutation_value_rate),
        init_population=initial_pop_lhs,
        objectives=partial(calculate_objectives_interp, train_model),
        crossover=crossover,
        mutation=mutation).solution(verbose=True)

    params = history.last().genotype

    if plot_figures:
        test_model = model_all_stations(forecasts_range=test_range)
        params = test_model.closest_params(params)
        closest_params = SWANParams(drf=params[0], cfw=params[1], stpm=params[2],
                                    fidelity_time=params[3], fidelity_space=params[4])

        drf_idx, cfw_idx, stpm_idx, fid_time_idx, fid_space_idx = test_model.params_idxs(closest_params)

        forecasts = test_model.grid[drf_idx, cfw_idx, stpm_idx, fid_time_idx, fid_space_idx]

        plot_results(forecasts=forecasts,
                     observations=wave_watch_results(path_to_results='../../samples/ww-res/', stations=ALL_STATIONS),
                     baseline=default_params_forecasts(test_model),
                     values_range=test_range)
        plot_population_movement(archive_history, grid)

    return history


if __name__ == '__main__':
    optimize_test(train_stations=[1, 2, 3], max_gens=10, pop_size=10, archive_size=5,
                  crossover_rate=0.7, mutation_rate=0.7, mutation_value_rate=[0.1, 0.01, 0.001])

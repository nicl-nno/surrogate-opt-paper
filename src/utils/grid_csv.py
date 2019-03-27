import csv
from math import sqrt

from src.basic_evolution.errors import (
    error_rmse_peak,
    error_rmse_all
)
from src.basic_evolution.model import (
    CSVGridFile,
    FidelityFakeModel
)
from src.basic_evolution.swan import SWANParams
from src.utils.files import (
    wave_watch_results
)


def grid_rmse():
    # fake, grid = real_obs_config()

    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')

    stations = [1, 2, 3]

    ww3_obs = \
        [obs.time_series() for obs in wave_watch_results(path_to_results='../../samples/ww-res/', stations=stations)]

    fake = FidelityFakeModel(grid_file=grid, observations=ww3_obs, stations_to_out=stations, error=error_rmse_peak,
                             forecasts_path='../../../wind-noice-runs/results_fixed/0')

    errors_total = []
    m_error = pow(10, 9)
    with open('../../samples/params_rmse.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['ID', 'DRF', 'CFW', 'STPM', 'RMSE_K1', 'RMSE_K2', 'RMSE_K3', 'TOTAL_RMSE'])
        for row in grid.rows:
            error = fake.output(params=row.model_params)
            print(grid.rows.index(row), error)

            if m_error > rmse(error):
                m_error = rmse(error)
                print(f"new min: {m_error}; {row.model_params.params_list()}")
            errors_total.append(rmse(error))

            row_to_write = row.model_params.params_list()
            row_to_write.extend(error)
            row_to_write.append(rmse(error))
            writer.writerow(row_to_write)

    print(f'min total rmse: {min(errors_total)}')


def rmse(vars):
    return sqrt(sum([pow(v, 2) for v in vars]) / len(vars))


def error_grid(noise_case=0):
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')
    stations = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    ww3_obs_all = \
        [obs.time_series() for obs in
         wave_watch_results(path_to_results='../../samples/ww-res/', stations=stations)]

    model_all = FidelityFakeModel(grid_file=grid, observations=ww3_obs_all, stations_to_out=stations,
                                  error=error_rmse_all,
                                  forecasts_path='../../../wind-postproc/out')

    with open(f'../../samples/params_rmse_{noise_case}.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        header = ['DRF', 'CFW', 'STPM']
        error_columns = [f'ERROR_K{station}' for station in stations]
        header.extend(error_columns)
        writer.writerow(header)

        fidelity = 240
        for row in grid.rows:
            metrics = model_all.output(
                params=SWANParams(drf=row.model_params.drf, cfw=row.model_params.cfw,
                                  stpm=row.model_params.stpm, fidelity_time=fidelity))
            row_to_write = row.model_params.params_list()
            row_to_write.extend(metrics)
            writer.writerow(row_to_write)


def all_error_grids():
    for noise_case in [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 25, 26]:
        error_grid(noise_case)


all_error_grids()

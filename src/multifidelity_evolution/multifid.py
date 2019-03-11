import os

import numpy as np

from src.basic_evolution.model import (
    FidelityFakeModel
)


class Multifid:
    def __init__(self, grid, fids, observations, path_to_forecasts, stations_to_out, error):
        '''
        Initialize an ensemble of FakeModels
        :param grid: CSVGridFile object with grid of model parameters
        :param noise_cases: List of noise case: 1 FakeModel instance per case
        :param observations: List of time series that correspond to observations
        :param path_to_forecasts: Directory where all noisy cases are located
        :param stations_to_out: List of station's indexes where to forecast
        '''

        self.grid = grid
        self.observations = observations
        self.forecasts_dir = path_to_forecasts
        self.stations_to_out = stations_to_out
        self.error = error
        self.models = self._initialized_models(fids=fids)

    def _initialized_models(self, fids):
        models = []
        for fid in fids:
            models.append(FidelityFakeModel(grid_file=self.grid, observations=self.observations,
                                            stations_to_out=self.stations_to_out, error=self.error,
                                            forecasts_path=os.path.join(self.forecasts_dir), fidelity=fid))

        return models

    def output(self, params):
        predictions = [model.output(params=params) for model in self.models]
        # predictions_ref = self.models[0].output(params=params)

        best_amount = 3
        predictions_best = sorted(predictions, key=lambda p: np.mean(p))[:best_amount]

        statistics_by_stations = {}
        for station_idx in range(len(self.stations_to_out)):
            predictions_for_station = column(predictions_best, station_idx)

            # predictions_coeff = 1 - abs(predictions_for_station / predictions_for_station[0] - 1)
            #
            # predictions_for_station = predictions_for_station * predictions_coeff + predictions_for_station[0] * (
            #         1 - predictions_coeff)

            statistics_by_stations[station_idx] = {'std': np.std(predictions_for_station),
                                                   'mean': np.mean(predictions_for_station),
                                                   'max': np.max(predictions_for_station)}

        out = []
        for station in statistics_by_stations.keys():
            mean = statistics_by_stations[station][
                'mean']  # *statistics_by_stations[station]['std']/np.std(predictions_ref)
            out.append(mean)

        return out

    def closest_params(self, params):
        drf = min(self.grid.drf_grid, key=lambda val: abs(val - params.drf))
        cfw = min(self.grid.cfw_grid, key=lambda val: abs(val - params.cfw))
        stpm = min(self.grid.stpm_grid, key=lambda val: abs(val - params.stpm))

        return drf, cfw, stpm


def column(matrix, idx):
    return [row[idx] for row in matrix]

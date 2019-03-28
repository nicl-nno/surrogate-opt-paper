from datetime import datetime

import numpy as np
from pyDOE import lhs
from pyKriging.krige import kriging
from scipy.stats.distributions import norm

from src.basic_evolution.swan import SWANParams

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


class KrigingModel:
    def __init__(self, grid_file, fake_model, station_idx, points_to_train, initial_fidelity):
        self.grid = grid_file
        self.model = fake_model
        self.station = station_idx
        self.fake_model = fake_model
        self.points_to_train = points_to_train

        self.fidelity = initial_fidelity

        self.features = self.features_from_lhs()

        self.krig = None

    def features_from_lhs(self):
        dim_num = 3
        samples_grid = lhs(dim_num, self.points_to_train, 'center')

        for idx, params_range in enumerate([self.grid.drf_grid, self.grid.cfw_grid, self.grid.stpm_grid]):
            samples_grid[:, idx] = norm(loc=np.mean(params_range), scale=np.std(params_range)).ppf(samples_grid[:, idx])

        features = [[drf, cfw, stpm] for drf, cfw, stpm in samples_grid]

        return np.asarray(features)

    def retrain_with_new_points(self, new_points):
        extended_features = self.features.tolist()
        for point in new_points:
            extended_features.append([point.drf, point.cfw, point.stpm])
            params = SWANParams(drf=point.drf, cfw=point.cfw, stpm=point.stpm,
                                fidelity_time=self.fidelity[0], fidelity_space=self.fidelity[1])
            print(self.fake_model.output_from_model(params=params)[0])

        self.points_to_train += len(new_points)

        print(f'retrain with new {len(new_points)} features')

        self.features = np.asarray(extended_features)
        self.train()

    def train(self, mode='lhs', **kwargs):
        target = []

        if mode is 'lhs':
            for feature in self.features:
                params = SWANParams(drf=feature[0], cfw=feature[1], stpm=feature[2],
                                    fidelity_time=self.fidelity[0], fidelity_space=self.fidelity[1])
                target.append(self.fake_model.output_from_model(params=params)[self.station])

        target = np.asarray(target)

        start_time = datetime.now().strftime(DATE_FORMAT)
        print(f'{start_time}: starting to train kriging model '
              f'with {self.points_to_train} points for station: {self.station}')

        krig = kriging(self.features, target, name='multikrieg')
        krig.train(optimizer='ga')
        self.krig = krig

        end_time = datetime.now().strftime(DATE_FORMAT)
        print(f'{end_time}: finished to train kriging model with'
              f' {self.points_to_train} points for station: {self.station}')

    def retrain_full(self, points, fidelity):
        self.fidelity = fidelity

        features = []
        for point in points:
            features.append([point.drf, point.cfw, point.stpm])

        print(f'retrain full with {len(points)} points with fidelity: {self.fidelity}')
        self.features = np.asarray(features)
        self.points_to_train = len(features)
        self.train()

    def prediction(self, params):
        assert self.krig is not None

        return self.krig.predict(params)

from datetime import datetime

import numpy as np
from pyKriging.krige import kriging

from src.basic_evolution.swan import SWANParams

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


class KrigingModel:
    def __init__(self, grid_file, fake_model, station_idx, points_to_train):
        self.grid = grid_file
        self.model = fake_model
        self.station = station_idx
        self.fake_model = fake_model

        self.features = self.features_from_grid()

        self.points_to_train = points_to_train
        self.krig = None

    def features_from_grid(self):
        features = []
        for drf in self.grid.drf_grid:
            for cfw in self.grid.cfw_grid:
                for stpm in self.grid.stpm_grid:
                    features.append([drf, cfw, stpm])

        return np.asarray(features)

    def train(self, mode='random', **kwargs):
        features = []
        target = []

        if mode is 'random':
            for _ in range(self.points_to_train):
                feature_idx = np.random.randint(0, len(self.features))
                features.append(self.features[feature_idx])

                params = SWANParams(drf=features[-1][0], cfw=features[-1][1], stpm=features[-1][2])
                target.append(self.fake_model.output_from_model(params=params)[self.station])

        features = np.asarray(features)
        target = np.asarray(target)

        start_time = datetime.now().strftime(DATE_FORMAT)
        print(f'{start_time}: starting to train kriging model for station: {self.station}')

        krig = kriging(features, target, name='multikrieg')
        krig.train(optimizer='ga')
        self.krig = krig

        end_time = datetime.now().strftime(DATE_FORMAT)
        print(f'{end_time}: finished to train kriging model for station: {self.station}')

    def prediction(self, params):
        assert self.krig is not None

        return self.krig.predict([params.drf, params.cfw, params.stpm])

from ast import literal_eval
from collections import Counter

import numpy as np

from src.evo_balt.evo import PhysicsType
from src.evo_balt.evo import SWANParams

OBSERVED_STATIONS = 3


class FakeModel:
    '''
    Class that imitates SWAN-model behaviour:
        it encapsulates simulation results on a model params grid:
            [drag, physics, wcr, ws] = model_output, i.e. forecasts
    '''

    def __init__(self, grid_file):
        '''
        Init parameters grid
        :param config: EvoConfig that contains parameters of evolution
        '''
        self.grid_file = grid_file
        self._init_grids()

    def _init_grids(self):
        self.grid = self._empty_grid()
        for row in self.grid_file.rows:
            drag_idx, physics_idx, wcr_idx, ws_idx = self.params_idxs(row.model_params)
            self.grid[drag_idx, physics_idx, wcr_idx, ws_idx] = \
                [FakeModel.Forecast(station_idx=idx, grid_row=row) for idx in range(OBSERVED_STATIONS)]

    def _empty_grid(self):
        return np.empty((len(self.grid_file.drag_grid), len(PhysicsType),
                         len(self.grid_file.wcr_grid),
                         len(self.grid_file.ws_grid)),
                        dtype=list)

    def params_idxs(self, params):
        drag_idx = self.grid_file.drag_grid.index(params.drag_func)
        physics_idx = (list(PhysicsType)).index(params.physics_type)
        wcr_idx = self.grid_file.wcr_grid.index(params.wcr)
        ws_idx = self.grid_file.ws_grid.index(params.ws)

        return drag_idx, physics_idx, wcr_idx, ws_idx

    def output(self, params):
        '''

        :param params: SWAN parameters
        :return: List of forecasts for each station
        '''
        drag_idx, physics_idx, wcr_idx, ws_idx = self.params_idxs(params=params)
        return self.grid[drag_idx, physics_idx, wcr_idx, ws_idx]

    class Forecast:
        def __init__(self, station_idx, grid_row):
            assert station_idx < len(grid_row.forecasts)

            self.station_idx = station_idx
            self.hsig_series = self._station_series(grid_row)

        def _station_series(self, grid_row):
            return grid_row.forecasts[self.station_idx]


class GridFile:
    '''
    Class that loads results of multiple SWAN simulations from CSV-file
    and construct grid of parameters
    '''

    def __init__(self, path):
        '''

        :param path: path to CSV-file
        '''
        self.path = path
        self._load()

    def _load(self):
        import csv
        with open(self.path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            grid_rows = [GridRow(row) for row in reader]
            self.rows = grid_rows

            drag_values = [row.model_params.drag_func for row in grid_rows]
            physics_types = [row.model_params.physics_type for row in grid_rows]
            wcr_values = [row.model_params.wcr for row in grid_rows]
            ws_values = [row.model_params.ws for row in grid_rows]

            self.drag_grid = self._grid_space(drag_values)
            self.wcr_grid = self._grid_space(wcr_values)
            self.ws_grid = self._grid_space(ws_values)
            self.physics_grid = self._grid_space(physics_types)

            print(self.drag_grid)
            print(self.wcr_grid)
            print(self.ws_grid)
            print(self.physics_grid)

    def _grid_space(self, param_values):
        '''
        Find parameter space of values counting all unique values
        '''

        cnt = Counter(param_values)
        return list(cnt.keys())


class GridRow:
    def __init__(self, row):
        self.id = row['ID']
        self.pop = row['Pop']
        self.error_distance = row['finErrorDist']
        self.model_params = self._swan_params(row['params'])
        self.errors = literal_eval(row['errors'])
        self.forecasts = literal_eval(row['forecasts'])

    def _swan_params(self, params_str):
        params_tuple = literal_eval(params_str)
        return SWANParams(drag_func=params_tuple[0], physics_type=params_tuple[1],
                          wcr=params_tuple[2], ws=params_tuple[3])


grid = GridFile(path="../../samples/grid_full.csv")
fake = FakeModel(grid_file=grid)
print(fake.output(params=SWANParams(drag_func=0.1, physics_type=PhysicsType.GEN3, wcr=0.4425, ws=0.00302)))

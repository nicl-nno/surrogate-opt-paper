from functools import partial

from src.basic_evolution.errors import (
    error_rmse_all
)
from src.basic_evolution.evo_operators import (
    calculate_objectives_interp
)
from src.basic_evolution.model import (
    CSVGridFile,
    FidelityFakeModel
)
from src.evolution.operators import default_operators
from src.evolution.spea2.dynamic import DynamicSPEA2
from src.evolution.spea2.spea2 import SPEA2
from src.multifidelity_evolution.fidelity_handler import FidelityHandler
from src.utils.files import (
    wave_watch_results
)


def run_evolution():
    train_stations = [1]
    grid = CSVGridFile('../../samples/wind-exp-params-new.csv')

    ww3_obs = \
        [obs.time_series() for obs in
         wave_watch_results(path_to_results='../../samples/ww-res/', stations=train_stations)]
    train_model = FidelityFakeModel(grid_file=grid, observations=ww3_obs,
                                    stations_to_out=train_stations, error=error_rmse_all,
                                    forecasts_path='../../../2fidelity/*', forecasts_range=(0, 1), sur_points=5,
                                    is_surrogate=True)

    operators = default_operators()

    handler = FidelityHandler(surrogates=train_model.surrogates_by_stations)

    _, _ = DynamicSPEA2(
        params=SPEA2.Params(max_gens=100, pop_size=10, archive_size=5,
                            crossover_rate=0.7, mutation_rate=0.7,
                            mutation_value_rate=[0.1, 0.01, 0.001]),
        objectives=partial(calculate_objectives_interp, train_model),
        evolutionary_operators=operators,
        fidelity_handler=handler).solution(verbose=True)


if __name__ == '__main__':
    run_evolution()

from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from src.utils.files import observations_from_range


def plot_results(forecasts, observations, **kwargs):
    fig, axs = plt.subplots(3, 3)
    time = np.linspace(1, len(forecasts[0].hsig_series), num=len(forecasts[0].hsig_series))

    obs_series = []
    for obs in observations:

        if 'values_range' in kwargs:
            obs_in_range = observations_from_range(
                obs.time_series(from_date="20140814.120000", to_date="20140915.000000"), kwargs['values_range'])
            obs_series.append(obs_in_range[:len(time)])
        else:
            obs_series.append(obs.time_series(from_date="20140814.120000", to_date="20140915.000000")[:len(time)])
    for idx in range(min(len(forecasts), len(observations))):
        i, j = divmod(idx, 3)
        station_idx = observations[idx].station_idx
        axs[i, j].plot(time, obs_series[idx],
                       label=f'Observations, Station {station_idx}')
        axs[i, j].plot(time, forecasts[idx].hsig_series,
                       label=f'Predicted, Station {station_idx}')
        if 'baseline' in kwargs:
            axs[i, j].plot(time, kwargs['baseline'][idx].hsig_series,
                           label=f'Baseline, Station {station_idx}')
        axs[i, j].legend()

    if 'save' in kwargs and kwargs['save'] is True:
        file_path = kwargs['file_path']
        fig.set_size_inches((11, 11), forward=False)
        fig.savefig(f'{file_path}.png', dpi=200)
        plt.close()
    else:
        plt.show()


def plot_population_movement(archive_history, grid):
    fig = plt.figure()
    ax = Axes3D(fig)
    max_history = []
    for pop in archive_history:

        drf = [individ.genotype.drf for individ in pop]
        cfw = [individ.genotype.cfw for individ in pop]
        stpm = [individ.genotype.stpm for individ in pop]

        pop_idx = archive_history.index(pop)

        rmse_values = []

        max_idx = -1
        for point_idx in range(len(pop)):
            rmse_val = rmse([obj for obj in pop[point_idx].objectives])
            rmse_values.append(rmse_val)
            max_idx = rmse_values.index(max(rmse_values))
        max_history.append([drf[max_idx], cfw[max_idx], stpm[max_idx]])

        for point_idx in range(len(pop)):
            color = 'red' if point_idx is not max_idx else 'blue'
            ax.scatter(drf[point_idx], cfw[point_idx], stpm[point_idx], c=color, s=5)
            ax.text(drf[point_idx], cfw[point_idx], stpm[point_idx], f'{rmse_val:.2f}({pop_idx})', size=10, zorder=1,
                    color='k')

    ax.plot(column(max_history, 0), column(max_history, 1), column(max_history, 2))
    ax.set_xlim(left=min(grid.drf_grid), right=max(grid.drf_grid))
    ax.set_ylim(bottom=min(grid.cfw_grid), top=max(grid.cfw_grid))
    ax.set_zlim(bottom=min(grid.stpm_grid), top=max(grid.stpm_grid))
    ax.set_xlabel('drf')
    ax.set_ylabel('cfw')
    ax.set_zlabel('stpm')
    plt.show()


def column(matrix, idx):
    return [row[idx] for row in matrix]


def rmse(vars):
    return sqrt(sum([pow(v, 2) for v in vars]) / len(vars))

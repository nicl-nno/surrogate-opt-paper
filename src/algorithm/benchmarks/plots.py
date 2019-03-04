from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from src.basic_evolution.model import (
    FakeModel,
    CSVGridFile
)
from src.basic_evolution.model import SWANParams


def population_variance_boxplots(history):
    variance = [p.pop_variance for p in history]
    plt.figure(figsize=(5, 5))
    sns.boxplot(
        data=variance,
    )
    plt.show()


def plot_pareto(pop):
    fig = plt.figure()

    ax = Axes3D(fig)
    ax.scatter([p.objectives[0] for p in pop], [p.objectives[1] for p in pop],
               [p.objectives[2] for p in pop])
    ax.set_xlabel('station 1')
    ax.set_ylabel('station 2')
    ax.set_zlabel('station 3')
    plt.show()


def plot_population_movement(pop, model):
    points = []

    drags = []
    wcrs = []
    errors = []
    for p in pop:
        drags.append(p.genotype.drag_func)
        wcrs.append(p.genotype.wcr)
        errors.append(p.fitness())

    x = np.zeros(len(model.grid_file.drag_grid))
    y = np.zeros(len(model.grid_file.wcr_grid))
    z = np.zeros((len(y), len(x)), dtype=np.float32)
    for drag_idx in range(len(model.grid_file.drag_grid)):
        for wcr_idx in range(len(model.grid_file.wcr_grid)):
            forecasts = model.grid[drag_idx, 1, wcr_idx, 0]
            rmse = \
                sqrt(pow(model.error_rmse_peak(forecasts[0]), 2) + pow(model.error_rmse_peak(forecasts[1]), 2) +
                     pow(model.error_rmse_peak(forecasts[2]), 2))

            points.append([model.grid_file.drag_grid[drag_idx], model.grid_file.wcr_grid[wcr_idx], rmse])
            x[drag_idx] = model.grid_file.drag_grid[drag_idx]
            y[wcr_idx] = model.grid_file.wcr_grid[wcr_idx]
            z[wcr_idx, drag_idx] = rmse

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()

    ax = Axes3D(fig)

    cset = ax.plot_surface(X, np.log10(Y), z, rstride=1, cstride=1,
                           cmap='viridis', edgecolor='none', alpha=0.7)

    ax.scatter(drags, np.log10(wcrs), errors, c='r', s=25)
    ax.clabel(cset, fontsize=9, inline=1)
    ax.set_xlabel('drag')
    ax.set_ylabel('wcr')
    ax.set_zlabel('rmse')
    plt.show()


def plot_params_space():
    grid = CSVGridFile('../../samples/wind-exp-params.csv')
    forecasts_path = '../../../samples/wind-noice-runs/results/1/'

    fake = FakeModel(grid_file=grid, forecasts_path=forecasts_path)

    drf, cfw = np.meshgrid(fake.grid_file.drf_grid, fake.grid_file.cfw_grid)
    stpm_fixed = fake.grid_file.stpm_grid[0]

    stations = 3
    error = np.ones(shape=(stations, drf.shape[0], drf.shape[1]))

    for i in range(drf.shape[0]):
        for j in range(drf.shape[1]):
            diff = fake.output(params=SWANParams(drf=drf[i, j], cfw=cfw[i, j], stpm=stpm_fixed))
            for station in range(stations):
                error[station, i, j] = diff[station]


def plot_error_variance(station_index=2, stmp_index=3):
    noise_dirs = {1: '../../../samples/wind-noice-runs/results/1/',
                  15: '../../../samples/wind-noice-runs/results/15/',
                  25: '../../../samples/wind-noice-runs/results/25/'}
    model_by_noise = {}

    grid = CSVGridFile('../../samples/wind-exp-params.csv')

    for noise in noise_dirs.keys():
        model = FakeModel(grid_file=grid, forecasts_path=noise_dirs[noise], fidelity=noise)
        model_by_noise[noise] = model

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for noise in model_by_noise.keys():
        drf, cfw, error = \
            params_and_error_grid(model=model_by_noise[noise], station_index=station_index, stpm_index=stmp_index)

        ax.scatter(drf, cfw, error, label=f'noise = {noise}')
        ax.set_xlabel('drf')
        ax.set_ylabel('cfw')

    plt.title(f'Error variance for K{station_index} station')
    plt.legend()
    plt.show()


def params_and_error_grid(model, station_index, stpm_index):
    drf, cfw = np.meshgrid(model.grid_file.drf_grid, model.grid_file.cfw_grid)
    stpm_fixed = model.grid_file.stpm_grid[stpm_index]

    error = np.ones(shape=drf.shape)
    for i in range(drf.shape[0]):
        for j in range(drf.shape[1]):
            out_by_stations = model.output(params=SWANParams(drf=drf[i, j], cfw=cfw[i, j], stpm=stpm_fixed))
            error[i, j] = out_by_stations[station_index]

    return drf, cfw, error


plot_error_variance()
# plot_params_space()

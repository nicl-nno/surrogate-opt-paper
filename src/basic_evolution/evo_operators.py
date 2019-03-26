import pickle
import random

import numpy as np
from pyDOE import lhs
from scipy.stats.distributions import norm

from src.basic_evolution.model import SWANParams

drf_range = [0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2, 1.4, 1.5999999999999999, 1.7999999999999998,
             1.9999999999999998, 2.1999999999999997, 2.4, 2.6, 2.8000000000000003]

cfw_range = [0.005, 0.01, 0.015, 0.02, 0.025, 0.030000000000000002, 0.035, 0.04, 0.045, 0.049999999999999996]
stpm_range = [0.001, 0.0025, 0.004, 0.0055, 0.006999999999999999, 0.008499999999999999, 0.009999999999999998]

fid_time_range = [60, 90, 120, 180]
fid_space_range = [14,28,56]

PARAMS = 5


def calculate_objectives(model, pop):
    '''
    Calculate two error functions i.e. |model_out - observation| ^ 2
    :param model: Class that can generate SWAN-like output for a given params
    :param pop: Population of SWAN-params i.e. individuals
    '''

    for p in pop:
        params = p.genotype
        closest = model.closest_params(params)
        params.update(drf=closest[0], cfw=closest[1], stpm=closest[2])
        p.objectives = tuple(model.output(params=params))


def calculate_objectives_interp(model, pop):
    '''
    Calculate two error functions i.e. |model_out - observation| ^ 2
    :param model: Class that can generate SWAN-like output for a given params
    :param pop: Population of SWAN-params i.e. individuals
    '''

    for p in pop:
        params = p.genotype
        p.objectives = tuple(model.output(params=params))


def crossover(p1, p2, rate):
    if random.random() >= rate:
        return p1

    part1_rate = abs(random.random())
    part2_rate = 1 - part1_rate

    child_params = SWANParams(drf=abs(p1.drf * part1_rate + p2.drf * part2_rate),
                              cfw=abs(p1.cfw * part1_rate + p2.cfw * part2_rate),
                              stpm=abs(p1.stpm * part1_rate + p2.stpm * part2_rate),
                              fidelity_space=abs(p1.fidelity_space * part1_rate + p2.fidelity_space * part2_rate),
                              fidelity_time=abs(p1.fidelity_time * part1_rate + p2.fidelity_time * part2_rate))
    return child_params


def mutation(individ, rate, mutation_value_rate):
    params = ['drf', 'cfw', 'stpm','fidelity_space','fidelity_time']
    if random.random() >= rate:
        param_to_mutate = params[random.randint(0, 4)]
        mutation_ratio = abs(np.random.RandomState().normal(1, 1.5, 1)[0])

        sign = 1 if random.random() < 0.5 else -1
        if param_to_mutate is 'drf':
            individ.drf += sign * mutation_value_rate[0] * mutation_ratio
            individ.drf = abs(individ.drf)
        if param_to_mutate is 'cfw':
            individ.cfw += sign * mutation_value_rate[1] * mutation_ratio
            individ.cfw = abs(individ.cfw)
        if param_to_mutate is 'stpm':
            individ.stpm += sign * mutation_value_rate[2] * mutation_ratio
            individ.stpm = abs(individ.stpm)
        if param_to_mutate is 'fidelity_space':
            individ.fidelity_space += sign * 7 * mutation_ratio
            individ.fidelity_space = abs(individ.fidelity_space)
        if param_to_mutate is 'fidelity_time':
            individ.fidelity_time += sign * 10 * mutation_ratio
            individ.fidelity_time = abs(individ.stpm)
    return individ


def default_initial_pop(size):
    return [SWANParams.new_instance() for _ in range(size)]


def initial_pop_lhs(size, **kwargs):
    samples_grid = lhs(PARAMS, size, 'center')
    for idx, params_range in enumerate([drf_range, cfw_range, stpm_range, fid_time_range, fid_space_range]):
        samples_grid[:, idx] = norm(loc=np.mean(params_range), scale=np.std(params_range)).ppf(samples_grid[:, idx])

    population = [SWANParams(drf=sample[0], cfw=sample[1], stpm=sample[2], fidelity_time=sample[3], fidelity_space=sample[4]) for sample in samples_grid]


    if 'dump' in kwargs and kwargs['dump'] is True:
        dump_population(population, kwargs['file_path'])

    return population


def initial_pop_lhs_from_file(file_path):
    with open(file_path, 'rb') as f:
        population = pickle.load(f)
        return population


def dump_population(population, file_path):
    pickle_out = open(file_path, 'wb')
    pickle.dump(population, pickle_out)
    pickle_out.close()

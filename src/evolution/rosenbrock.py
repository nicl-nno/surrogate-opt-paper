import random
from datetime import datetime

import numpy as np

from src.evolution.operators import EvoOperators
from src.evolution.spea2 import DefaultSPEA2


def calculate_objectives_rosenbrock(pop):
    for p in pop:
        x, y = p.genotype
        p.objectives = tuple([rosenbrook(x, y)])


def rosenbrook(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def initial_pop_rosenbrook(size):
    random.seed(42)
    return [[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(size)]


def crossover(p1, p2, rate):
    random.seed(datetime.now())
    if random.random() <= rate:
        return p1

    x = p1[0] if random.randint(0, 1) else p2[0]
    y = p1[0] if random.randint(0, 1) else p2[0]

    return [x, y]


def mutation(individ, rate, mutation_value_rate):
    random.seed(datetime.now())
    if random.random() <= rate:
        idx = random.randint(0, 1)
        sign = 1 if random.random() < 0.5 else -1
        mutation_ratio = abs(np.random.normal(1, 5, 1)[0])
        individ[idx] += sign * mutation_value_rate[idx] * mutation_ratio

        return individ
    else:
        return individ


def print_best_rosenbrook(best, gen_index):
    print(f"new best: {best.fitness() : .3f}, x : {best.genotype[0]}, y : {best.genotype[1]}, f = {best.objectives}")
    print(gen_index)


def rosenbrook_optimize_test():
    operators = EvoOperators(init_population=initial_pop_rosenbrook, crossover=crossover, mutation=mutation)
    history, _ = DefaultSPEA2(
        params=DefaultSPEA2.Params(max_gens=500, pop_size=10, archive_size=5,
                                   crossover_rate=0.7, mutation_rate=0.7,
                                   mutation_value_rate=[0.05, 0.05]),
        objectives=calculate_objectives_rosenbrock,
        evolutionary_operators=operators).solution(verbose=True, print_fun=print_best_rosenbrook)


if __name__ == '__main__':
    rosenbrook_optimize_test()

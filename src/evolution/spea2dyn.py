import copy
import random
from math import sqrt
from operator import itemgetter

import numpy as np

from src.evolution.raw_fitness import raw_fitness


class SPEA2Dyn:
    def __init__(self, params, init_population, objectives, crossover, mutation, initial_fidelity, delta_fidelity):
        '''
         Strength Pareto Evolutionary Algorithm
        :param params: Meta-parameters of the SPEA2
        :param init_population: function to generate initial population
        :param objectives: function to calculate objective functions for each individual in population
        :param crossover: function to crossover two genotypes
        :param mutation: function to mutate genotype
        '''
        self.params = params

        self.init_population = init_population
        self.objectives = objectives
        self.crossover = crossover
        self.mutation = mutation
        self.init_population=init_population

       # self._init_populations()


    def solution(self, verbose=True, **kwargs):
        fid_ref_freq=3
        new_fidelity=initial_fidelity
        for i in range(round(self.params.max.max_gens/fid_ref_freq)):
            history, archive_history = SPEA2(
                params=self.params,
                init_population=self.init_population(history),
                objectives=self.objectives,
                crossover=self.crossover,
                mutation=self.mutation,
                fidelity=new_fidelity.solution(verbose=verbose)
            new_fidelity=new_fidelity-delta_fidelity

        return history, archive_history

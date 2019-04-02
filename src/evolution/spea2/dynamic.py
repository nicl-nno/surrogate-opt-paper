import copy

from src.basic_evolution.model import (
    SWANPerfModel
)
from .default import (
    mean_obj,
    print_new_best_individ,
    rmse
)
from .spea2 import SPEA2


class DynamicSPEA2(SPEA2):
    def __init__(self, params, objectives, evolutionary_operators, fidelity_handler, **kwargs):
        super().__init__(params=params, objectives=objectives, evolutionary_operators=evolutionary_operators)
        self.handler = fidelity_handler

        if 'points_by_fidelity' in kwargs:
            self.points_by_fidelity = kwargs['points_by_fidelity']
        else:
            self.points_by_fidelity = {}

    def solution(self, verbose=True, **kwargs):
        archive_history = []
        history = SPEA2.ErrorHistory()

        gen = 0
        self.handler.init(population=self._archive + self._pop)

        while gen < self.params.max_gens:
            self.fitness()
            self._archive = self.environmental_selection(self._pop, self._archive)
            best = sorted(self._archive, key=lambda p: mean_obj(p))[0]

            last_fit = history.last().fitness_value
            if last_fit > mean_obj(best):
                best_gens = best.genotype

                if verbose:
                    if 'print_fun' in kwargs:
                        kwargs['print_fun'](best, gen)
                    else:
                        print_new_best_individ(best, gen)

                history.add_new(best_gens, gen, mean_obj(best),
                                rmse(best))

                self.handler.handle_new_min_found(population=self._archive, gen_idx=gen)

            selected = self.selected(self.params.pop_size, self._archive)
            self._pop = self.reproduce(selected, self.params.pop_size)

            to_add = copy.deepcopy(self._archive + self._pop)
            self.objectives(to_add)
            archive_history.append(to_add)

            self.handler.handle_new_generation(population=self._archive + self._pop, gen_idx=gen,
                                               points_by_fidelity=self.points_by_fidelity)
            gen += 1

        return history, archive_history


class DynamicSPEA2PerfModel:

    def get_execution_time(sur_points, initial_fidelity, params, handler):

        def sur_execution_time(num_points):
            return 0.0001 * num_points ** 2.5

        num_gens = params.max_gens
        pop_size = params.pop_size

        num_points_for_retrain = handler.point_for_retrain
        gens_to_change_fidelity = handler.gens_to_change_fidelity
        time_delta = handler.time_delta
        space_delta = handler.space_delta

        min_found_prob = 0.3

        execution_time_init = (SWANPerfModel.get_execution_time(initial_fidelity)) * sur_points + sur_execution_time(
            sur_points)

        gens_frame = int(round(num_gens / gens_to_change_fidelity))

        execution_time_add_runs = 0
        execution_time_retrain_runs = (SWANPerfModel.get_execution_time(
            initial_fidelity)) * num_points_for_retrain * gens_to_change_fidelity * min_found_prob

        current_fidelity = initial_fidelity

        for i in range(gens_frame):
            if (current_fidelity[0] - time_delta > 0 and current_fidelity[1] - space_delta > 0):
                current_fidelity = (current_fidelity[0] - time_delta, current_fidelity[1] - space_delta)
                execution_time_retrain_runs += (SWANPerfModel.get_execution_time(
                    current_fidelity)) * pop_size / 3 + sur_execution_time(pop_size / 3)
            execution_time_add_runs += SWANPerfModel.get_execution_time(
                current_fidelity) * num_points_for_retrain + sur_execution_time(
                num_points_for_retrain) * gens_to_change_fidelity * min_found_prob

        execution_time = execution_time_init + execution_time_add_runs + execution_time_retrain_runs

        return execution_time

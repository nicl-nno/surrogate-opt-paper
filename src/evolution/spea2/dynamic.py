import copy

from .default import (
    mean_obj,
    print_new_best_individ,
    rmse
)
from .spea2 import SPEA2


class DynamicSPEA2(SPEA2):
    def __init__(self, params, objectives, evolutionary_operators, fidelity_handler):
        super().__init__(params=params, objectives=objectives, evolutionary_operators=evolutionary_operators)
        self.handler = fidelity_handler

    def solution(self, verbose=True, **kwargs):
        archive_history = []
        history = SPEA2.ErrorHistory()

        gen = 0
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

                self.handler.handle(population=self._archive, gen_idx=gen)

            selected = self.selected(self.params.pop_size, self._archive)
            self._pop = self.reproduce(selected, self.params.pop_size)

            to_add = copy.deepcopy(self._archive + self._pop)
            self.objectives(to_add)
            archive_history.append(to_add)

            gen += 1

        return history, archive_history

fidelity_time = [60, 90, 120, 180]
fidelity_space = [14, 28, 56]

MIN_FID_TIME = 60
MIN_FID_SPACE = 14

from itertools import product

from src.basic_evolution.swan import SWANParams
from src.evolution.spea2.default import mean_obj


class FidelityHandler:
    def __init__(self, surrogates, time_delta, space_delta, point_for_retrain, gens_to_change_fidelity):
        self.surrogates = surrogates
        self.time_delta = time_delta
        self.space_delta = space_delta
        self.point_for_retrain = point_for_retrain
        self.last_min_at_gen = -1
        self.gens_to_change_fidelity = gens_to_change_fidelity

    def init(self, population):
        initial_fidelity = self.surrogates[0].fidelity
        print(f'initial fid: {initial_fidelity}')
        self.set_fidelity(population=population, new_fidelity=self.surrogates[0].fidelity)
        self.train_surrogates(fidelity=initial_fidelity)

    def train_surrogates(self, fidelity, **kwargs):
        # fidelity = self.surrogates[0].fidelity
        if 'points_to_train' in kwargs:
            for model in self.surrogates:
                model.train_with_mixed_points(fidelity=fidelity, external_points=kwargs['points_to_train'])
        else:
            for model in self.surrogates:
                model.train_with_mixed_points(fidelity=fidelity)

    def handle_new_min_found(self, population, gen_idx):
        self.last_min_at_gen = gen_idx

        print(f'starting to retrain at generation: {gen_idx}')

        new_points = self.best_individuals(population=population)

        for model in self.surrogates:
            model.retrain_with_new_points(new_points=new_points)

    def handle_new_generation(self, population, gen_idx, **kwargs):
        if self.last_min_at_gen != -1 and self.__gens_after_last_min(gen_idx) >= self.gens_to_change_fidelity:
            current_fid_time = population[0].genotype.fid_time
            current_fid_space = population[0].genotype.fid_space

            new_fidelity = self.__next_fidelity(current_fidelity=(current_fid_time, current_fid_space))

            self.set_fidelity(population=population, new_fidelity=new_fidelity)

            if (current_fid_time, current_fid_space) != new_fidelity:
                self.last_min_at_gen = gen_idx
                print(f'fidelity has been changed at {gen_idx} generation:'
                      f' {(current_fid_time, current_fid_space)} -> {new_fidelity}')

                if 'points_by_fidelity' in kwargs:
                    external_points = self.external_points(new_fidelity, kwargs['points_by_fidelity'])
                    print(f'external points: {len(external_points)}')
                    points_for_train = self.__extracted_points(population) + external_points
                    self.train_surrogates(points_to_train=points_for_train, fidelity=new_fidelity)

    def external_points(self, fidelity, points_by_fidelity):
        points = []
        if fidelity in points_by_fidelity:
            points = points_by_fidelity[fidelity]

        return points

    def set_fidelity(self, population, new_fidelity):
        fid_time, fid_space = new_fidelity
        for individ in population:
            individ.genotype.fid_time = fid_time
            individ.genotype.fid_space = fid_space

    def retrain_models_with_new_fidelity(self, points, fidelity):
        new_points = [point.genotype for point in points]
        for model in self.surrogates:
            model.retrain_full(points=new_points, fidelity=fidelity)

    def best_individuals(self, population):
        assert self.point_for_retrain <= len(population)

        best = sorted(population, key=lambda p: mean_obj(p))[:self.point_for_retrain]

        return [individ.genotype for individ in best]

    def __gens_after_last_min(self, gen_idx):
        return gen_idx - self.last_min_at_gen

    def __next_fidelity(self, current_fidelity):

        current_fid_time, current_fid_space = current_fidelity
        new_fid_time = max(MIN_FID_TIME, current_fid_time - self.time_delta)
        new_fid_space = max(MIN_FID_SPACE, current_fid_space - self.space_delta)

        return new_fid_time, new_fid_space

    def __extracted_points(self, population):
        return [individ.genotype for individ in population]


def default_points_by_fidelity(size=10):
    points_by_fidelity = {}

    for fidelity in fidelity_combinations():
        points = []
        for _ in range(size):
            params = SWANParams.new_instance()
            params.fid_time = fidelity[0]
            params.fid_space = fidelity[1]

            points.append(params)

        points_by_fidelity[fidelity] = points

    return points_by_fidelity


def fidelity_combinations():
    fidelity = []
    for fid_time, fid_space in product(fidelity_time, fidelity_space):
        fidelity.append((fid_time, fid_space))

    return fidelity

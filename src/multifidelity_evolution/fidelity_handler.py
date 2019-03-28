fidelity_time = [60, 90, 120, 180]
fidelity_space = [14, 28, 56]

from src.evolution.spea2.default import mean_obj


class FidelityHandler:
    def __init__(self, surrogates):
        self.point_for_retrain = 3
        self.surrogates = surrogates
        self.last_min_at_gen = -1
        self.gens_to_change_fidelity = 10

    def init_fidelity(self, population):
        self.set_fidelity(population=population, new_fidelity=(60, 14))

    def handle_new_min_found(self, population, gen_idx):
        self.last_min_at_gen = gen_idx

        print(f'starting to retrain at generation: {gen_idx}')

        new_points = self.best_individuals(population=population)

        for model in self.surrogates:
            model.retrain_with_new_points(new_points=new_points)

    def handle_new_generation(self, population, gen_idx):
        if self.last_min_at_gen != -1 and self.__gens_after_last_min(gen_idx) >= self.gens_to_change_fidelity:
            current_fid_time = population[0].genotype.fid_time
            current_fid_space = population[0].genotype.fid_space

            new_fidelity = self.__next_fidelity(current_fidelity=(current_fid_time, current_fid_space))

            print(f'fidelity has been changed: {(current_fid_time, current_fid_space)} -> {new_fidelity}')

            self.set_fidelity(population=population, new_fidelity=new_fidelity)

            if (current_fid_time, current_fid_space) != new_fidelity:
                self.retrain_models_with_new_fidelity(points=population, fidelity=new_fidelity)

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
        current_fid_time_idx = fidelity_time.index(current_fidelity[0])
        current_fid_space_idx = fidelity_space.index(current_fidelity[1])

        new_fid_time, new_fid_space = fidelity_time[current_fid_time_idx], fidelity_space[current_fid_space_idx]

        if current_fid_time_idx < len(fidelity_time) - 1:
            new_fid_time = fidelity_time[current_fid_time_idx + 1]

        if current_fid_space_idx < len(fidelity_space) - 1:
            new_fid_space = fidelity_space[current_fid_space_idx + 1]

        return new_fid_time, new_fid_space

fidelity_time = [60, 90, 120, 180]
fidelity_space = [14, 28, 56]

from src.evolution.spea2.default import mean_obj


class FidelityHandler:
    def __init__(self, surrogates):
        self.point_for_retrain = 10
        self.surrogates = surrogates

    def handle(self, population, gen_idx):
        print(f'starting to retrain at generation: {gen_idx}')

        new_points = self.best_individuals(population=population)

        for model in self.surrogates:
            model.retrain_with_new_points(new_points=new_points)

    def change_fidelity(self, population, new_fidelity):
        fid_time, fid_space = new_fidelity
        for individ in population:
            individ.genotype.fid_time = fid_time
            individ.genotype.fid_space = fid_space

    def best_individuals(self, population):
        assert self.point_for_retrain <= len(population)

        best = sorted(population, key=lambda p: mean_obj(p))[:self.point_for_retrain]

        return [individ.genotype for individ in best]

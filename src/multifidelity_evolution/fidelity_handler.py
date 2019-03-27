import random

fidelity_time = [60, 90, 120, 180]
fidelity_space = [14, 28, 56]


class FidelityHandler:
    def __init__(self):
        pass

    def handle(self, population, gen_idx):
        # Change fidelity of all population by random each 10 generation
        if gen_idx % 10 == 0:
            new_fidelity = (random.choice(fidelity_time), random.choice(fidelity_space))
            self.change_fidelity(population, new_fidelity=new_fidelity)
            print(f'fidelity has been changed to : {new_fidelity}')

    def change_fidelity(self, population, new_fidelity):
        fid_time, fid_space = new_fidelity
        for individ in population:
            individ.genotype.fid_time = fid_time
            individ.genotype.fid_space = fid_space

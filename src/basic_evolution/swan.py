drf_range = [0.2, 0.4, 0.6000000000000001, 0.8, 1.0, 1.2, 1.4, 1.5999999999999999, 1.7999999999999998,
             1.9999999999999998, 2.1999999999999997, 2.4, 2.6, 2.8000000000000003]

cfw_range = [0.005, 0.01, 0.015, 0.02, 0.025, 0.030000000000000002, 0.035, 0.04, 0.045, 0.049999999999999996]
stpm_range = [0.001, 0.0025, 0.004, 0.0055, 0.006999999999999999, 0.008499999999999999, 0.009999999999999998]

import random


class SWANParams:

    @staticmethod
    def new_instance():
        return SWANParams(drf=random.choice(drf_range), cfw=random.choice(cfw_range), stpm=random.choice(stpm_range))

    def __init__(self, drf, cfw, stpm, fidelity_time=60, fidelity_space=14):
        self.drf = drf
        self.cfw = cfw
        self.stpm = stpm
        self.fid_time = fidelity_time
        self.fid_space = fidelity_space

    def update(self, drf, cfw, stpm, fidelity_time, fidelity_space):
        self.drf = drf
        self.cfw = cfw
        self.stpm = stpm
        self.fid_time = fidelity_time
        self.fid_space = fidelity_space

    def params_list(self):
        return [self.drf, self.cfw, self.stpm, self.fid_time, self.fid_space]

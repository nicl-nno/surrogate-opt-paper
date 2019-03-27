from .spea2 import SPEA2


class DynamicSPEA2(SPEA2):
    def solution(self, verbose=True, **kwargs):
        raise NotImplementedError

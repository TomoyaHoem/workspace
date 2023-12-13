import numpy as np
import pymoo.core


class Alg:
    """Surrogate algorithm object to not rely on deep copy"""

    def __init__(self, pop: list, n_gen: int, opt: np.array) -> None:
        self.pop = pop
        self.n_gen = n_gen
        self.opt = opt
        self.problem = None


class AlgorithmResult:
    """
    Custom result object to store pymoo results.
    Emulate same data container but prevent need for
    deep copy of history to save memory.
    """

    def __init__(self, name: str, F: np.array, X: np.array, algs: list[Alg]) -> None:
        self.name = name
        self.F = F
        self.X = X
        self.history = algs

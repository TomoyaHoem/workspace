import math
import numpy as np

"""
Class to store average data over multiple runs for one algorithm
alg_name: Algorithm name (NSGA2, NSGA3 or MOEAD)
pareto: List of number of pareto entries per run
qed, logp, sa: List of three Lists that store the min, max and average objective values per run
running_data: List of Lists containing the full (over all generations) running metric data per run
"""


class AlgData:
    def __init__(self, alg_name) -> None:
        self.alg_name = alg_name
        self._pareto_len = []
        self._pareto = []
        self._qed = [[], [], []]
        self._logp = [[], [], []]
        self._sa = [[], [], []]
        self._running_data = []
        self._histories = []
        self._hv = []

    @property
    def histories(self):
        return self._histories

    @histories.setter
    def histories(self, value):
        self._histories.append(value)

    @property
    def pareto_len(self):
        return math.floor(np.mean(self._pareto_len))

    @pareto_len.setter
    def pareto_len(self, value: int):
        self._pareto_len.append(value)

    @property
    def pareto(self):
        return self._pareto

    @pareto.setter
    def pareto(self, value: list):
        self._pareto.append(value)

    @property
    def hv(self):
        return self._hv

    @pareto.setter
    def hv(self, value: list):
        self._hv.append(value)

    @property
    def qed(self):
        return np.round(np.mean(self._qed, axis=1), 2)

    @qed.setter
    def qed(self, values: list):
        self._qed = np.hstack((self._qed, values))

    @property
    def logp(self):
        return np.round(np.mean(self._logp, axis=1), 2)

    @logp.setter
    def logp(self, values: list):
        self._logp = np.hstack((self._logp, values))

    @property
    def sa(self):
        return np.round(np.mean(self._sa, axis=1), 2)

    @sa.setter
    def sa(self, values: list):
        self._sa = np.hstack((self._sa, values))

    @property
    def running_data(self):
        vals = [v[0][2] for v in self._running_data]
        min_data = [
            self.alg_name + "_min",
            self._running_data[0][0][1],
            np.min(vals, axis=0),
            self._running_data[0][0][3],
        ]
        max_data = [
            self.alg_name + "_max",
            self._running_data[0][0][1],
            np.max(vals, axis=0),
            self._running_data[0][0][3],
        ]
        avg_data = [
            self.alg_name + "_avg",
            self._running_data[0][0][1],
            np.mean(vals, axis=0),
            self._running_data[0][0][3],
        ]
        r_data = {
            "MIN": min_data,
            "MAX": max_data,
            "AVG": avg_data,
        }

        return r_data

    @running_data.setter
    def running_data(self, values: list):
        self._running_data.append(values)

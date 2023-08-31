import numpy as np

X1, X2 = np.meshgrid(np.linspace(-2, 2, 500), np.linspace(-2, 2, 500))

F1 = 100 * (X1**2 + X2**2)
F2 = (X1 - 1) ** 2 + X2**2

G1 = 2 * (X1[0] - 0.1) * (X1[0] - 0.9)
G2 = 40 * (X1[0] - 0.4) * (X1[0] - 0.6)

import matplotlib.pyplot as plt

plt.rc("font", family="serif")

levels = np.array([0.02, 0.1, 0.25, 0.5, 0.8])
plt.figure(figsize=(7, 5))
CS = plt.contour(X1, X2, F1, 10 * levels, colors="black", alpha=0.5)
CS.collections[0].set_label("$f_1(x)$")

CS = plt.contour(X1, X2, F2, levels, linestyles="dashed", colors="black", alpha=0.5)
CS.collections[0].set_label("$f_2(x)$")

plt.plot(X1[0], G1, linewidth=2.0, color="green", linestyle="dotted")
plt.plot(X1[0][G1 < 0], G1[G1 < 0], label="$g_1(x)$", linewidth=2.0, color="green")

plt.plot(X1[0], G2, linewidth=2.0, color="blue", linestyle="dotted")
plt.plot(
    X1[0][X1[0] > 0.6], G2[X1[0] > 0.6], label="$g_2(x)$", linewidth=2.0, color="blue"
)
plt.plot(X1[0][X1[0] < 0.4], G2[X1[0] < 0.4], linewidth=2.0, color="blue")

plt.plot(np.linspace(0.1, 0.4, 100), np.zeros(100), linewidth=3.0, color="orange")
plt.plot(np.linspace(0.6, 0.9, 100), np.zeros(100), linewidth=3.0, color="orange")

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

plt.legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, fancybox=True, shadow=False
)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))

f2 = lambda f1: -(((f1 / 100) ** 0.5 - 1) ** 2)
F1_a, F1_b = np.linspace(1, 16, 300), np.linspace(36, 81, 300)
F2_a, F2_b = f2(F1_a), f2(F1_b)

plt.rc("font", family="serif")
plt.plot(F1_a, F2_a, linewidth=2.0, color="green", label="Pareto-front")
plt.plot(F1_b, F2_b, linewidth=2.0, color="green")

plt.xlabel("$f_1$")
plt.ylabel("$f_2$")

plt.legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=4, fancybox=True, shadow=False
)

plt.tight_layout()
plt.show()

import numpy as np
from pymoo.core.problem import ElementwiseProblem


class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=2, n_obj=2, n_ieq_constr=2, xl=np.array([-2, -2]), xu=np.array([2, 2])
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 100 * (x[0] ** 2 + x[1] ** 2)
        f2 = (x[0] - 1) ** 2 + x[1] ** 2

        g1 = 2 * (x[0] - 0.1) * (x[0] - 0.9) / 0.18
        g2 = -20 * (x[0] - 0.4) * (x[0] - 0.6) / 4.8

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


problem = MyProblem()

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True,
)

from pymoo.termination import get_termination

termination = get_termination("n_gen", 40)

from pymoo.optimize import minimize

res = minimize(problem, algorithm, termination, seed=1, save_history=True, verbose=True)

X = res.X
F = res.F

import matplotlib.pyplot as plt

xl, xu = problem.bounds()
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], s=30, facecolors="none", edgecolors="r")
plt.xlim(xl[0], xu[0])
plt.ylim(xl[1], xu[1])
plt.title("Design Space")
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors="none", edgecolors="blue")
plt.title("Objective Space")
plt.show()

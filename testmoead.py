from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

problem = get_problem("WFG7", n_var=5, n_obj=3)

ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)

algorithm = MOEAD(
    ref_dirs,
    n_neighbors=15,
    prob_neighbor_mating=0.7,
)

res = minimize(problem, algorithm, ("n_gen", 200), seed=1, verbose=True)

Scatter().add(res.F).show()

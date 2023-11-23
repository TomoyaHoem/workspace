import time
import os
import sys
import random

from result_processor import ResultProcessor
from averagewriter import AverageWriter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import selfies as sf

from rdkit import Chem
from rdkit.Chem import QED

from rdkit.Chem import RDConfig

from guacamole_tasks import Task

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # type: ignore

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from moead_div import MOEAD

from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.util.ref_dirs import get_reference_directions


alphabet = sf.get_semantic_robust_alphabet()

SEED = 1
NUM_ITERATIONS = 10  # 200
POP_SIZE = 100
REPEAT = 1  # 10


class SELFIESProblem(ElementwiseProblem):
    def __init__(self, selfies, task: Task):
        self.task = task
        super().__init__(n_var=1, n_obj=self.task.num_obj, n_ieq_constr=0)
        self.SELFIES = selfies

    def _evaluate(self, x, out, *args, **kwargs):
        # decode SELFIES individual to SMILES
        mol = Chem.MolFromSmiles(sf.decoder(x[0]))

        # add QED and SA objective, invert QED to minimize
        # add guacamole task objectives, invert to minimize
        objectives = [-QED.default(mol), sascorer.calculateScore(mol)] + [
            -obj.score_mol(mol) for obj in self.task()
        ]

        out["F"] = np.array(objectives, dtype=float)


class SEFLIESSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1), None, dtype=object)

        np.random.seed(SEED)
        sample = np.random.choice(problem.SELFIES, size=n_samples, replace=False)

        for i in range(n_samples):
            X[i, 0] = sample[i]

        return X


class SELFIESCrossover(Crossover):
    def __init__(self):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)

    def onepoint(self, parents: list) -> list:
        split_parent_one = list(sf.split_selfies(parents[0]))
        split_parent_two = list(sf.split_selfies(parents[1]))

        cut_point_one = (
            len(split_parent_one) // 2
        )  # random.randint(0, len(split_parent_one))
        cut_point_two = (
            len(split_parent_two) // 2
        )  # random.randint(0, len(split_parent_two))

        parent_one_cut_one = split_parent_one[0:cut_point_one]
        parent_one_cut_two = split_parent_one[cut_point_one : len(split_parent_one)]

        parent_two_cut_one = split_parent_two[0:cut_point_two]
        parent_two_cut_two = split_parent_two[cut_point_two : len(split_parent_two)]

        child_one = "".join(parent_one_cut_one + parent_two_cut_two)
        child_two = "".join(parent_one_cut_two + parent_two_cut_one)

        return [child_one, child_two]

    def _do(self, problem, X, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=object)

        # for each mating provided
        for k in range(n_matings):
            # get the first and the second parent
            a, b = X[0, k, 0], X[1, k, 0]

            # prepare the offsprings
            offspring = self.onepoint([a, b])

            # join the character list and set the output
            Y[0, k, 0], Y[1, k, 0] = offspring[0], offspring[1]

        return Y


class SELFIESMutation(Mutation):
    def __init__(self):
        super().__init__()

    def mutate(self, sfi: str) -> str:
        selfie_split = list(sf.split_selfies(sfi))

        rnd_symbol = random.sample(list(alphabet), 1)[0]
        rnd_ind = random.randint(0, len(selfie_split) - 1)
        # print(f"Replacing random symbol at: {rnd_ind}, with: {rnd_symbol}")

        selfie_split[rnd_ind] = rnd_symbol

        return "".join(selfie_split)

    def _do(self, problem, X, **kwargs):
        # for each individual
        for i in range(len(X)):
            r = np.random.random()

            # with a probabilty of 40% - replace one random token
            if r < 0.4:
                try:
                    X[i, 0] = self.mutate(X[i, 0])
                except Exception as e:
                    print(e)
                    print(X[i, 0])

        return X


class SELFIESDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return a.X[0] == b.X[0]


# TODO: fix
def print_help():
    """
    Arguments: Data Algorithm1 Algorithm2
    """
    print("")


def main(
    args: list,
    mols: pd.DataFrame,
):  # aw: AverageWriter) -> None:
    print("Passed args: ", end=" ")
    print(*args)
    print(f"Popsize: {POP_SIZE}, Iterations: {NUM_ITERATIONS}")
    print("")

    molecules = mols

    # * I. Parse algorithms

    print("Parsing Algorithms...")

    if len(args) < 4:
        print(
            "ERROR: invalid number of arguments please provide <[Algorithms] Filename Options Task>."
        )
        return

    # unpack arguments
    algs, filename, store_print, guac = args
    task = Task(guac)

    algorithms = []

    for alg in algs:
        print(f"Read {alg.upper()}")
        if alg == "nsga2":
            # run pymoo nsga2
            algorithm = NSGA2(
                pop_size=POP_SIZE,
                sampling=SEFLIESSampling(),
                crossover=SELFIESCrossover(),
                mutation=SELFIESMutation(),
                eliminate_duplicates=SELFIESDuplicateElimination(),
            )
        elif alg == "nsga3":
            # create the reference directions to be used for the optimization
            ref_dirs = get_reference_directions("energy", task.num_obj, POP_SIZE)
            # run pymoo nsga3
            algorithm = NSGA3(
                ref_dirs=ref_dirs,
                sampling=SEFLIESSampling(),
                crossover=SELFIESCrossover(),
                mutation=SELFIESMutation(),
                eliminate_duplicates=SELFIESDuplicateElimination(),
            )
        elif alg == "moead":
            ref_dirs = get_reference_directions("energy", task.num_obj, POP_SIZE)
            # run pymoo moead
            algorithm = MOEAD(
                ref_dirs=ref_dirs,
                n_neighbors=15,
                prob_neighbor_mating=0.7,
                sampling=SEFLIESSampling(),
                crossover=SELFIESCrossover(),
                mutation=SELFIESMutation(),
            )
        else:
            print(f"ERROR: invalid argument: {alg}")
            return
        algorithms.append(algorithm)

    # * II. Run Algorithms

    results = []
    sets = {
        "Data": "ZINC20-Subset",
        "Task": guac,
        "Seed": SEED,
        "Pop_size": POP_SIZE,
        "N_Gen": NUM_ITERATIONS,
        "Sampling": "Random uniform",
        "Crossover": "1-point, 100%",
        "Mutation": "Random replace, 40%",
    }

    for alg_n, alg in zip(algs, algorithms):
        r = run_alg(molecules, alg, alg_n, task)

        # multiply negative objectives by -1 since they were minimized
        # all except at index i = 1 because SA score should remain minimized
        r.F = np.array(
            [[-v if i != 1 else v for i, v in enumerate(indiv)] for indiv in r.F]
        )

        results.append(r)

    # * III. Store Results

    # rw = ResultWriter(molecules, results, sets, args[2], SEED)
    rp = ResultProcessor(molecules, results, task, sets, filename)
    rp(store_print)
    # aw.append_results(results)


def run_alg(molecules, algorithm, alg: str, task: Task):
    print(f"Running {alg.upper()}...")
    res = minimize(
        SELFIESProblem(selfies=molecules["SELFIES"].to_numpy(), task=task),
        algorithm,
        ("n_gen", NUM_ITERATIONS),
        # seed=SEED,
        save_history=True,
        verbose=True,
    )

    print(f"Finished {alg.upper()}")
    print("")

    return res


def read_data(data: str):
    # * II. Parse data
    print(f"Reading *{data}* Data...")

    # load data
    start = time.time()

    if data == "fragments":
        # unpickle
        molecules = pd.read_pickle("./pkl/1%-fragments-indicators-lfs.pkl")

        # add a column to separate pareto front
        molecules["pareto"] = "#9C95994C"
    elif data == "druglike":
        # unpickle
        molecules = pd.read_pickle("./pkl/1%-druglike-indicators-lfs.pkl")

        # add a column to separate pareto front
        molecules["pareto"] = "#9C95994C"
    elif data == "subset":
        # unpickle
        molecules = pd.read_pickle("./pkl/subset-selfies.pkl")
    else:
        print("ERROR: invalid data name")
        return

    end = time.time()
    dur = round(end - start, 3)
    print(f"Elapsed time to unpickle: {dur}s")
    print(molecules.head())
    print("")

    return molecules


if __name__ == "__main__":
    """
    Datasets: subset
    Algorithms: NSGA2, NSGA3, MOEAD
    Store/Print options: -p, -s or -ps / -sp
    """
    # Entry
    print("Pymoo MOP using SELFIES")
    print("# " * 10)
    print("")
    # Settings
    pop_sizes = [100]  # , 500]
    algs = ["nsga2", "nsga3"]  # , "moead"]
    tasks = [
        "Cobimetinib"
    ]  # , "Fexofenadine", "Osimertinib", "Pioglitazone", "Ranolazine"]
    store_print = "-s"
    repeat = REPEAT
    # Read Data
    input_mols = read_data("subset")
    # Run
    r_count, i_count = 0, 0
    print("Starting runs...")
    print("-" * 25)
    print("")
    for t in tasks:
        for p in pop_sizes:
            # aw = AverageWriter(algs)
            for i in range(repeat):
                POP_SIZE = p
                filename = (
                    "MOP_Experiment_"
                    + str(r_count)
                    + "_"
                    + "_".join(algs)
                    + "_"
                    + str(NUM_ITERATIONS)
                    + "_"
                    + str(POP_SIZE)
                    + ".xlsx"
                )
                r_count += 1
                main([algs, filename, store_print, t], input_mols)  # , aw)
                print(f"Finished run {r_count}")
                print("-" * 25)
                print("")
            print("Storing Averages...")
            # aw.store_averages(
            #     "MOP_Experiment_Averages_"
            #     + str(i_count)
            #     + "_"
            #     + str(repeat)
            #     + "_"
            #     + "_".join(algs)
            #     + "_"
            #     + str(NUM_ITERATIONS)
            #     + "_"
            #     + str(POP_SIZE)
            #     + ".xlsx",
            #     repeat,
            # )
            i_count += 1
            print("")

    print("")
    print("# " * 10)
    print("Finished Execution")

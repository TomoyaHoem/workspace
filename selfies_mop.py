import time
import os
import sys
import random

from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import selfies as sf

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen

from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # type: ignore

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD

from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter

alphabet = sf.get_semantic_robust_alphabet()


class SELFIESProblem(ElementwiseProblem):
    def __init__(self, selfies):
        super().__init__(n_var=1, n_obj=3, n_ieq_constr=0)
        self.SELFIES = selfies

    def _evaluate(self, x, out, *args, **kwargs):
        qed, logp, sa = 0, 0, 0

        mol = Chem.MolFromSmiles(sf.decoder(x[0]))

        qed = QED.default(mol)
        logp = Crippen.MolLogP(mol)
        try:
            sa = sascorer.calculateScore(mol)
        except Exception as e:
            print(e)
            print(x[0])
            print(sf.decoder(x[0]))

        out["F"] = np.array([-qed, -logp, sa], dtype=float)


class SEFLIESSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1), None, dtype=object)

        sample = np.random.choice(problem.SELFIES, size=n_samples)

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


def main() -> None:
    print("Pymoo MOP using SELFIES")
    print("#" * 10)
    print(f"Running {sys.argv[1]} \n")

    # load data
    start = time.time()
    # unpickle
    molecules = pd.read_pickle("./pkl/100-fragments-indicators.pkl")

    # add selfies representation
    molecules["SELFIES"] = molecules["Smiles"].apply(sf.encoder)

    end = time.time()
    dur = round(end - start, 3)
    print(f"Elapsed time to unpickle and add SELFIES: {dur}s")
    print(molecules.head())

    alg = sys.argv[1]

    if alg == "nsga2":
        # run pymoo nsga2
        algorithm = NSGA2(
            pop_size=100,
            sampling=SEFLIESSampling(),
            crossover=SELFIESCrossover(),
            mutation=SELFIESMutation(),
            eliminate_duplicates=SELFIESDuplicateElimination(),
        )
    elif alg == "nsga3":
        # create the reference directions to be used for the optimization
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=18)
        # run pymoo nsga3
        algorithm = NSGA3(
            pop_size=190,
            ref_dirs=ref_dirs,
            sampling=SEFLIESSampling(),
            crossover=SELFIESCrossover(),
            mutation=SELFIESMutation(),
            eliminate_duplicates=SELFIESDuplicateElimination(),
        )
    elif alg == "moead":
        ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
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
        print("ERROR: invalid argument")
        return

    res = minimize(
        SELFIESProblem(selfies=molecules["SELFIES"].to_numpy()),
        algorithm,
        ("n_gen", 30),
        seed=1,
        verbose=True,
    )

    # add a column to separate pareto front
    molecules["pareto"] = "#9C95994C"

    results = res.X[np.argsort(res.F[:, 0])]
    print(np.column_stack(results))

    # maximize objectives
    for obj_vals in res.F:
        obj_vals[0] *= -1
        obj_vals[1] *= -1

    mol_sample = molecules.sample(frac=0.1, random_state=1)

    # add results to dataframe
    for mol, obj in zip(res.X, res.F):
        new_row = pd.DataFrame(
            {
                "Dir": "",
                "File": "",
                "Mol": "",
                "SMILES": "",
                "QED": obj[0],
                "LogP": obj[1],
                "SA": obj[2],
                "SELFIES": mol,
                "pareto": "#FF0022FF",
            }
        )
        mol_sample = pd.concat([mol_sample, new_row], axis=0, ignore_index=True)

    print(mol_sample.head(-10))

    # plot data
    # Creating figure
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter(
        mol_sample["QED"],
        mol_sample["LogP"],
        mol_sample["SA"],
        color=mol_sample["pareto"],
    )
    ax.set_xlabel("QED", fontweight="bold")
    ax.set_ylabel("LogP", fontweight="bold")
    ax.set_zlabel("SA", fontweight="bold")
    plt.title("MOP Result")

    # show plot
    plt.show()

    Scatter().add(res.F).show()


if __name__ == "__main__":
    main()

import time
import os
import sys

import pandas as pd
import numpy as np

import selfies as sf

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen

from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # type: ignore

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination


class SELFIESProblem(ElementwiseProblem):
    def __init__(self, selfies):
        super().__init__(n_var=1, n_obj=3, n_ieq_constr=0)
        self.SELFIES = selfies

    def _evaluate(self, x, out, *args, **kwargs):
        qed, logp, sa = 0, 0, 0

        mol = Chem.MolFromSmiles(sf.decoder(x[0]))

        qed = QED.default(mol)
        logp = Crippen.MolLogP(mol)
        sa = sascorer.calculateScore(mol)

        out["F"] = np.array([qed, logp, sa], dtype=float)


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
            off_a = ["_"] * problem.n_characters
            off_b = ["_"] * problem.n_characters

            for i in range(problem.n_characters):
                if np.random.random() < 0.5:
                    off_a[i] = a[i]
                    off_b[i] = b[i]
                else:
                    off_a[i] = b[i]
                    off_b[i] = a[i]

            # join the character list and set the output
            Y[0, k, 0], Y[1, k, 0] = "".join(off_a), "".join(off_b)

        return Y


class SELFIESMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        # for each individual
        for i in range(len(X)):
            r = np.random.random()

            # with a probabilty of 40% - change the order of characters
            if r < 0.4:
                perm = np.random.permutation(problem.n_characters)
                X[i, 0] = "".join(np.array([e for e in X[i, 0]])[perm])

            # also with a probabilty of 40% - change a character randomly
            elif r < 0.8:
                prob = 1 / problem.n_characters
                mut = [
                    c
                    if np.random.random() > prob
                    else np.random.choice(problem.ALPHABET)
                    for c in X[i, 0]
                ]
                X[i, 0] = "".join(mut)

        return X


class SELFIESDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return a.X[0] == b.X[0]


def main() -> None:
    print("NSGA2 in pymoo using SELFIES")

    # load data
    start = time.time()
    # unpickle
    molecules = pd.read_pickle("./pkl/100-fragments.pkl")

    # add selfies representation
    molecules["SELFIES"] = molecules["Smiles"].apply(sf.encoder)

    end = time.time()
    dur = round(end - start, 3)
    print(f"Elapsed time to unpickle and add SELFIES: {dur}s")
    print(molecules.head())

    # run pymoo nsga2
    algorithm = NSGA2(
        pop_size=100,
        sampling=SEFLIESSampling(),
        crossover=SELFIESCrossover(),
        mutation=SELFIESMutation(),
        eliminate_duplicates=SELFIESDuplicateElimination(),
    )

    res = minimize(
        SELFIESProblem(selfies=molecules["SELFIES"].to_numpy()),
        algorithm,
        ("n_gen", 100),
        seed=1,
        verbose=True,
    )

    # plot results


if __name__ == "__main__":
    main()

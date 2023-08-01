# 1. Random one-point crossover (or n-point), repeat until valid found

# 2. Truncate randomly from left and right, concatenate truncations, fix parantheses, repeat until valid

# 3. Consider rings, non-rings

# 4. Include valence rules

# 5. Predefine collection of constraints
import time
import pandas as pd
import random
import os
import sys

from rdkit import Chem
from rdkit import RDLogger


def test_crossover(molecules: pd.DataFrame, crossover: callable, fileno: int) -> None:
    print(f"Testing crossover method: {crossover.__name__}")
    test_count = 5000
    fail = 0

    for i in range(test_count):
        parents = molecules["Smiles"].sample(n=2, random_state=1)
        children = crossover(parents.values.tolist())
        for c in children:
            mol = Chem.MolFromSmiles(c)
            if not mol:
                os.write(fileno, b"\n")
                fail += 1

    success = round((1 - fail / (test_count * 2)) * 100, 2)

    print(
        f"{success}% ({test_count * 2 - fail} / {test_count * 2}) crossovers succeeded!"
    )


# one-point crossover
# produces two childs from random diagonal splits of parents
def onepoint(parents: list) -> list:
    cut_point_one = random.randint(0, len(parents[0]))
    cut_point_two = random.randint(0, len(parents[1]))

    parent_one_cut_one = parents[0][0:cut_point_one]
    parent_one_cut_two = parents[0][cut_point_one : len(parents[0])]

    parent_two_cut_one = parents[1][0:cut_point_two]
    parent_two_cut_two = parents[1][cut_point_two : len(parents[0])]

    child_one = parent_one_cut_one + parent_two_cut_two
    child_two = parent_one_cut_two + parent_two_cut_one

    return [child_one, child_two]


def main() -> None:
    start = time.time()
    # unpickle
    molecules = pd.read_pickle("./pkl/100-fragments-pareto.pkl")
    end = time.time()
    dur = round(end - start, 3)

    print(f"Elapsed time to unpickle: {dur}s")
    print(molecules.head())

    # get a sample of smiles
    # parents = molecules["Smiles"].sample(n=10, random_state=1)
    # print(parents.head())

    # RDLogger.DisableLog("rdApp.*")
    stderr_fileno = sys.stderr.fileno()
    stderr_save = os.dup(stderr_fileno)
    # file descriptor of log file
    stderr_fd = open("error.log", "w")
    os.dup2(stderr_fd.fileno(), stderr_fileno)

    # try crossover variants

    # 1.
    test_crossover(molecules, onepoint, stderr_fd.fileno())

    # close the log file
    stderr_fd.close()
    # restore old sys err
    os.dup2(stderr_save, stderr_fileno)

    # RDLogger.EnableLog("rdApp.*")


if __name__ == "__main__":
    main()

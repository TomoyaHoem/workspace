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
import re

from rdkit import Chem
from rdkit import RDLogger


# https://github.com/knu-chem-lcbc/molfinder-v0/blob/main/ModSmi.py
# molfinder method that returns list of indices to avoid when splitting smiles string
def set_avoid_ring(_smiles):
    avoid_ring = []
    ring_tmp = set(re.findall(r"\d", _smiles))
    for j in ring_tmp:
        tmp = [i for i, val in enumerate(_smiles) if val == j]
        while tmp:
            avoid_ring += [j for j in range(tmp.pop(0), tmp.pop(0) + 1)]
    return set(avoid_ring)


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

    return onepoint_cross(parents, cut_point_one, cut_point_two)


def onepoint_cross(parents, cut_point_one, cut_point_two):
    parent_one_cut_one = parents[0][0:cut_point_one]
    parent_one_cut_two = parents[0][cut_point_one : len(parents[0])]

    parent_two_cut_one = parents[1][0:cut_point_two]
    parent_two_cut_two = parents[1][cut_point_two : len(parents[0])]

    child_one = parent_one_cut_one + parent_two_cut_two
    child_two = parent_one_cut_two + parent_two_cut_one

    return [child_one, child_two]


def onepoint_avoidring(parents: list) -> list:
    parent_one_avoid_set = set_avoid_ring(parents[0])
    parent_two_avoid_set = set_avoid_ring(parents[1])

    cutpoints_parent_one = set(range(len(parents[0]))).difference(parent_one_avoid_set)
    cutpoints_parent_two = set(range(len(parents[1]))).difference(parent_two_avoid_set)

    if len(cutpoints_parent_one) == 0 or len(cutpoints_parent_two) == 0:
        print("No way to avoid ring")
        return ["", ""]

    cut_point_one = random.sample(tuple(cutpoints_parent_one), 1)
    cut_point_two = random.sample(tuple(cutpoints_parent_two), 1)

    return onepoint_cross(parents, cut_point_one[0], cut_point_two[0])


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

    stderr_fileno = sys.stderr.fileno()
    stderr_save = os.dup(stderr_fileno)
    # file descriptor of log file
    stderr_fd = open("error.log", "w")
    os.dup2(stderr_fd.fileno(), stderr_fileno)

    # try crossover variants

    # 1.
    test_crossover(molecules, onepoint_avoidring, stderr_fd.fileno())

    # close the log file
    stderr_fd.close()
    # restore old sys err
    os.dup2(stderr_save, stderr_fileno)


if __name__ == "__main__":
    main()

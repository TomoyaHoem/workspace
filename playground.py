import pandas as pd
import numpy as np
import re
from rdkit import Chem


def main() -> None:
    m = Chem.MolFromSmiles("O=[N+]O-H")
    if m is None:
        print("Invalid Molecule")
        return


if __name__ == "__main__":
    main()

    # _smiles = "C1CCCCC1C2CCCCC2"
    # avoid_ring = []
    # ring_tmp = set(re.findall(r"\d", _smiles))
    # print(ring_tmp)
    # for j in ring_tmp:
    #     tmp = [i for i, val in enumerate(_smiles) if val == j]
    #     while tmp:
    #         avoid_ring += [j for j in range(tmp.pop(0), tmp.pop(0) + 1)]
    # print(set(avoid_ring))

    # a = set(range(16))

    # print(a.difference(avoid_ring))

    # print(re.findall(r"[^(]", _smiles)[0])

    # 0 3 - 0 6 - 3 6
    # 1 4 - 1 7 - 4 7
    # 2 5 - 2 8 - 5 8

    # for i in range(3):
    #     for j in range(3):
    #         if j > 1:
    #             k = 3 + i
    #             l = 6 + i
    #         else:
    #             k = i
    #             l = 3 + i + 3 * j
    #         print(k, l)

    # a = [1, 2, 3, 4]
    # b = [5, 6, 7, 8]

    # c = [a, b]
    # end = False
    # currentList = a
    # index = 1

    # while not end:
    #     for i, x in enumerate(currentList):
    #         print(f"i: {i}, x: {x}")
    #         if i == len(a) - 1:
    #             if index >= len(c):
    #                 print("end")
    #                 end = True
    #                 break
    #             currentList = c[index]
    #             index += 1

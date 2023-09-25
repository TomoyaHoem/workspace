import pandas as pd
import numpy as np
import re
from rdkit import Chem
import random
import time
import selfies as sf
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit import DataStructs

alphabet = sf.get_semantic_robust_alphabet()


def onepoint(parents: list) -> list:
    split_parent_one = list(sf.split_selfies(parents[0]))
    split_parent_two = list(sf.split_selfies(parents[1]))

    cut_point_one = random.randint(0, len(split_parent_one))
    cut_point_two = random.randint(0, len(split_parent_two))

    parent_one_cut_one = split_parent_one[0:cut_point_one]
    parent_one_cut_two = split_parent_one[cut_point_one : len(split_parent_one)]

    parent_two_cut_one = split_parent_two[0:cut_point_two]
    parent_two_cut_two = split_parent_two[cut_point_two : len(split_parent_two)]

    child_one = "".join(parent_one_cut_one + parent_two_cut_two)
    child_two = "".join(parent_one_cut_two + parent_two_cut_one)

    return [child_one, child_two]


def mutate(sfi: str) -> str:
    r = np.random.random()

    selfie_split = list(sf.split_selfies(sfi))

    rnd_symbol = random.sample(list(alphabet), 1)[0]
    rnd_ind = random.randint(0, len(selfie_split) - 1)
    print(f"Replacing random symbol at: {rnd_ind}, with: {rnd_symbol}")

    selfie_split[rnd_ind] = rnd_symbol

    return "".join(selfie_split)


def main() -> None:
    mol = "[C][C][C][C][#S][C][=C][C][=P][C][=C][Ring1][=Branch1][=Branch2][C][C][C][C][=C][C][=C][C][=C][Ring1][=Branch1][C][=C]"
    smiles = sf.decoder(mol)

    print(smiles)

    # # load data
    # start = time.time()
    # # unpickle
    # molecules = pd.read_pickle("./pkl/100-fragments-indicators.pkl")
    # print(molecules["QED"].max())
    # print(molecules["QED"].min())

    # print(molecules["LogP"].max())
    # print(molecules["LogP"].min())

    # print(molecules["SA"].max())
    # print(molecules["SA"].min())

    # # add selfies representation
    # molecules["SELFIES"] = molecules["Smiles"].apply(sf.encoder)

    # end = time.time()
    # dur = round(end - start, 3)
    # print(f"Elapsed time to unpickle and add SELFIES: {dur}s")
    # print(molecules.head())

    # mol = molecules["SELFIES"].sample(n=1).iloc[0]
    # print(f"Before: {mol}")
    # mol_mut = mutate(mol)
    # print(f"After: {mol_mut}")

    # # mol_prev = Chem.MolFromSmiles(sf.decoder(mol))
    # # mol_new = Chem.MolFromSmiles(sf.decoder(mol_mut))

    # # img_prev = Draw.MolToImage(mol_prev)
    # # img_prev.show()
    # # img_new = Draw.MolToImage(mol_new)
    # # img_new.show()

    # parents = molecules["SELFIES"].sample(n=2, random_state=1)
    # children = onepoint(parents.values.tolist())

    # print("Parents:")

    # for parent in parents:
    #     print(sf.decoder(parent))
    #     mol = Chem.MolFromSmiles(sf.decoder(parent))
    #     img = Draw.MolToImage(mol)
    #     img.show()

    # print("")
    # print("Children:")

    # for child in children:
    #     print(sf.decoder(child))
    #     mol = Chem.MolFromSmiles(sf.decoder(child))
    #     img = Draw.MolToImage(mol)
    #     img.show()


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

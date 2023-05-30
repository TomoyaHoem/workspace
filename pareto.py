from rdkit import Chem
import os
import numpy as np


def main() -> None:
    print(f"Entry")

    # read all smiles strings
    cwd = os.getcwd() + "\ZINCSMILES"

    molecules = []

    for dirnum, (root, dirs, files) in enumerate(os.walk(cwd)):
        for dir in dirs:
            molecules.append([])

        for filenum, file in enumerate(files):
            if dirnum > 1:
                break
            molecules[dirnum - 1].append([])
            curP = root + "\\" + file
            suppl = Chem.SmilesMolSupplier(curP)
            for molnum, mol in enumerate(suppl):
                print(f"Storing SMILES: {molnum} of {curP}")
                molecules[dirnum - 1][filenum - 1].append(mol)

    # evaluate all molecules for logp, sa, qed

    # plot result

    # apply non-dominated sorting

    # plot pareto front


if __name__ == "__main__":
    main()

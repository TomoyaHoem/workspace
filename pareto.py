from rdkit import Chem
import os


def main() -> None:
    print(f"Entry")

    # read all smiles strings
    cwd = os.getcwd() + "\ZINCSMILES"

    molecules = []

    for dirnum, (root, dirs, files) in enumerate(os.walk(cwd)):
        if dirs:
            for dir in dirs:
                print(dir)
                molecules.append(dir)

    print(molecules)

    # for filenum, file in enumerate(files):
    #    if file.endswith("smi"):
    #        suppl = Chem.SmilesMolSupplier(root + "\\" + file)
    #        for mol in suppl:
    #            molecules[dirnum][filenum].append(mol)

    # evaluate all molecules for logp, sa, qed

    # plot result

    # apply non-dominated sorting

    # plot pareto front


if __name__ == "__main__":
    main()

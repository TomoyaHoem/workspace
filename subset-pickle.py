import os
import pandas as pd
from alive_progress import alive_bar

TOTAL_NUM_MOLS = 435_000_000
MOL_PERCENTAGE_PER_FILE = 0.001


def read_molecules(cwd: str) -> pd.DataFrame:
    dictionary_list = []

    with alive_bar(int(MOL_PERCENTAGE_PER_FILE * TOTAL_NUM_MOLS)) as bar:
        for root, dirs, files in os.walk(cwd):
            for file in files:
                index = 0
                curP = os.path.join(root, file)
                num_lines = sum(1 for _ in open(curP))
                for line in open(curP):
                    if index > int(MOL_PERCENTAGE_PER_FILE * num_lines):
                        # print("LIMIT")
                        break
                    smile = line
                    mol_row = {
                        "Smiles": smile,
                    }
                    # print(f"SMILES no {index} + {curMol}")
                    dictionary_list.append(mol_row)
                    index += 1
                    bar()

    print(f"Stored a total of {len(dictionary_list)} molecules")
    return pd.DataFrame.from_dict(dictionary_list)


def main() -> None:
    print("Reading Subset molecules to pkl....")
    # read data from files into dataframe
    cwd = os.path.join(os.getcwd(), "Subset")

    molecules = read_molecules(cwd)

    # print info including memory size
    print(molecules.info())

    print("Dataframe Head")
    print(molecules.head())

    # pickle dataframe
    print("pickle...")
    molecules.to_pickle("./pkl/subset.pkl")
    print("--- Finished Pickling ---")


if __name__ == "__main__":
    main()

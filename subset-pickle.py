import os
import time
import pandas as pd
from alive_progress import alive_bar

TOTAL_NUM_MOLS = 435_000_000
MOL_PERCENTAGE_PER_FILE = 0.1


def set_d_types(df):
    df["Smiles"] = df["Smiles"].astype("string")
    return df


def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b:
            break
        yield b


def read_molecules(cwd: str) -> pd.DataFrame:
    dictionary_list = []

    with alive_bar(int(MOL_PERCENTAGE_PER_FILE * TOTAL_NUM_MOLS)) as bar:
        start = time.time()
        for root, dirs, files in os.walk(cwd):
            for file in files:
                index = 0
                curP = os.path.join(root, file)
                with open(curP, "r", encoding="utf-8", errors="ignore") as f:
                    num_lines = sum(bl.count("\n") for bl in blocks(f))
                with open(curP) as f:
                    next(f)
                    for line in f:
                        if index > int(MOL_PERCENTAGE_PER_FILE * num_lines):
                            # print("LIMIT")
                            break
                        smile = line.split()[0]
                        mol_row = {
                            "Smiles": smile,
                        }
                        # print(f"SMILES no {index} + {curMol}")
                        dictionary_list.append(mol_row)
                        index += 1
                        bar()

    end = time.time()
    dur = end - start
    print(f"Elapsed time to read SMILES data: {dur}s")

    print(f"Stored a total of {len(dictionary_list)} molecules")
    return pd.DataFrame.from_dict(dictionary_list)


def main() -> None:
    print("Reading Subset molecules to pkl....")
    # read data from files into dataframe
    cwd = os.path.join(os.getcwd(), "Subset")

    molecules = read_molecules(cwd)

    # cast smiles to string for memory efficiency
    print(molecules.info())
    molecules = set_d_types(molecules)
    print(molecules.info())

    print("Dataframe Head")
    print(molecules.head())

    # pickle dataframe
    print("pickle...")
    molecules.to_pickle("./pkl/subset-lfs.pkl")
    print("--- Finished Pickling ---")


if __name__ == "__main__":
    main()

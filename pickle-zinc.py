import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
import selfies as sf
from alive_progress import alive_bar

TOTAL_NUM_MOLS = 15_000_000
MOL_PERCENTAGE_PER_FILE = 0.03


def SetDTypes(df):
    df["Dir"] = df["Dir"].astype("category")
    df["File"] = df["File"].astype("category")
    return df


def ReadAllMols(cwd, molecules) -> None:
    count = 0

    with alive_bar(TOTAL_NUM_MOLS) as bar:
        for root, dirs, files in os.walk(cwd):
            for file in files:
                curP = root + "\\" + file
                suppl = Chem.SmilesMolSupplier(curP)
                for mol in suppl:
                    curMol = [root[-2:], file, mol, Chem.MolToSmiles(mol)]
                    molecules.loc[count] = curMol
                    count += 1
                    bar()

    print(f"Stored a total of {count} molecules")


def ReadMolsWithLimit(cwd) -> pd.DataFrame:
    dictionary_list = []

    with alive_bar(int(MOL_PERCENTAGE_PER_FILE * TOTAL_NUM_MOLS)) as bar:
        for root, dirs, files in os.walk(cwd):
            for file in files:
                index = 0
                # print(f"Storing {file}")
                curP = root + "\\" + file
                suppl = Chem.SmilesMolSupplier(curP)
                num_lines = sum(1 for _ in open(curP))
                # if num_lines == 0:
                #     print(f"Skipping {file} because empty.")
                #     continue
                for mol in suppl:
                    if index > int(MOL_PERCENTAGE_PER_FILE * num_lines):
                        # print("LIMIT")
                        break
                    smile = Chem.MolToSmiles(mol)
                    mol_row = {
                        "Dir": root[-2:],
                        "File": file,
                        # "Mol": mol,
                        "Smiles": smile,
                        "SELFIES": sf.encoder(smile),
                    }
                    # print(f"SMILES no {index} + {curMol}")
                    dictionary_list.append(mol_row)
                    index += 1
                    bar()

    print(f"Stored a total of {len(dictionary_list)} molecules")
    return pd.DataFrame.from_dict(dictionary_list)


def main() -> None:
    print(f"--- Reading ZINC data ---")

    # create dataframe
    molecules = pd.DataFrame(columns=["Dir", "File", "Mol", "Smiles"])

    # read data from files into dataframe
    cwd = os.getcwd() + "\Fragments"
    # current index to store in dataframe
    molecules = ReadMolsWithLimit(cwd)

    # downcast directory and filename to category to save memory
    print(molecules.info())
    molecules = SetDTypes(molecules)
    print(molecules.info())

    # add molecule image column to dataframe, does not work currently?
    # PandasTools.AddMoleculeColumnToFrame(molecules, "Smiles", "Molecule")

    print("Dataframe Head")
    print(molecules.head())

    # pickle dataframe
    molecules.to_pickle("./pkl/10%-fragments.pkl")
    print("--- Finished Pickling ---")


if __name__ == "__main__":
    main()

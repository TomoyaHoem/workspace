import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from alive_progress import alive_bar

TOTAL_NUM_MOLS = 1300000
MOL_100_PER_FILE = 7943


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


def ReadMolsWithLimit(cwd, maxFromFile, molecules) -> None:
    count = 0

    with alive_bar(MOL_100_PER_FILE) as bar:
        for root, dirs, files in os.walk(cwd):
            for file in files:
                index = 0
                # print(f"Storing {file}")
                curP = root + "\\" + file
                suppl = Chem.SmilesMolSupplier(curP)
                for mol in suppl:
                    if index > maxFromFile:
                        # print("LIMIT")
                        break
                    curMol = [root[-2:], file, mol, Chem.MolToSmiles(mol)]
                    # print(f"SMILES no {index} + {curMol}")
                    molecules.loc[count] = curMol
                    count += 1
                    index += 1
                    bar()

    print(f"Stored a total of {count} molecules")


def main() -> None:
    print(f"--- Reading ZINC data ---")

    # create dataframe
    molecules = pd.DataFrame(columns=["Dir", "File", "Mol", "Smiles"])

    # read data from files into dataframe
    cwd = os.getcwd() + "\ZINCSMILES"
    # current index to store in dataframe
    maxFromFile = 100
    ReadMolsWithLimit(cwd, maxFromFile, molecules)

    # downcast directory and filename to category to save memory
    # print(molecules.info())
    # molecules = SetDTypes(molecules)
    # print(molecules.info())

    # add molecule column to dataframe, does not work currently?
    # PandasTools.AddMoleculeColumnToFrame(molecules, "Smiles", "Molecule")

    print("Dataframe Head")
    print(molecules.head())

    # pickle dataframe
    molecules.to_pickle("./pkl/100-shards.pkl")
    print("--- Finished Pickling ---")


if __name__ == "__main__":
    main()

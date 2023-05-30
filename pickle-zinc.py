import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools


def SetDTypes(df):
    df["Dir"] = df["Dir"].astype("category")
    df["File"] = df["File"].astype("category")
    return df


def ReadMols(cwd, maxFromFile, molecules) -> None:
    index = 0

    for root, dirs, files in os.walk(cwd):
        for file in files:
            # print(f"Storing {file}")
            curP = root + "\\" + file
            suppl = Chem.SmilesMolSupplier(curP)
            for mol in suppl:
                if index != 0 and index % maxFromFile == 0:
                    # print("LIMIT")
                    break
                curMol = [root[-2:], file, mol, Chem.MolToSmiles(mol)]
                # print(f"SMILES no {index} + {curMol}")
                molecules.loc[index] = curMol
                index += 1


def main() -> None:
    print(f"--- Pickling ZINC data ---")

    # create dataframe
    molecules = pd.DataFrame(columns=["Dir", "File", "Mol", "Smiles"])

    # read data from files into dataframe
    cwd = os.getcwd() + "\ZINCSMILES"
    # current index to store in dataframe
    maxFromFile = 1000
    ReadMols(cwd, maxFromFile, molecules)

    # downcast directory and filename to category to save memory
    print(molecules.info())
    molecules = SetDTypes(molecules)
    print(molecules.info())

    # add molecule column to dataframe, does not work currently?
    # PandasTools.AddMoleculeColumnToFrame(molecules, "Smiles", "Molecule")

    print("Dataframe Head")
    print(molecules.head())

    # pickle dataframe
    molecules.to_pickle("./pkl/shards_test.pkl")


if __name__ == "__main__":
    main()

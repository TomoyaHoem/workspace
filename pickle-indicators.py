import sys
import os
import gc
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen

from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # type: ignore


def indicators(molecules: pd.DataFrame) -> pd.DataFrame:
    print("Starting next batch...")

    # get mol object from smiles
    start = time.time()
    molecules["Mol"] = molecules["Smiles"].apply(Chem.MolFromSmiles)
    end = time.time()
    dur = round(end - start, 3)
    print(f"Elapsed time to get mol object: {dur}s")

    # evaluate all molecules for qed, log, sa
    start = time.time()
    molecules["QED"] = molecules["Mol"].apply(QED.default)
    end = time.time()
    dur = round(end - start, 3)
    print(f"Elapsed time to calculate QED: {dur}s")

    start = time.time()
    molecules["LogP"] = molecules["Mol"].apply(Crippen.MolLogP)
    end = time.time()
    dur = round(end - start, 3)
    print(f"Elapsed time to calculate LogP: {dur}s")

    start = time.time()
    molecules["SA"] = molecules["Mol"].apply(sascorer.calculateScore)
    end = time.time()
    dur = round(end - start, 3)
    print(f"Elapsed time to calculate SA score: {dur}s")

    molecules = molecules.drop(columns=["Mol"])

    return molecules


def main() -> None:
    print(f"Entry")

    start = time.time()
    # unpickle
    molecules = pd.read_pickle("./pkl/filtred-subset-lfs.pkl")
    print(molecules.head())
    end = time.time()
    dur = round(end - start, 3)

    print(f"Elapsed time to unpickle: {dur}s")

    dfs = np.array_split(molecules, 3)
    mol_dfs = []
    for i, df in enumerate(dfs):
        mol_dfs.append(indicators(df))
        if (i+1) % 10 == 0:
            molecules = pd.concat(mol_dfs)
            # pickle result
            print("Pickle...")
            molecules.to_pickle("./pkl/subset-indicators.pkl_" + str(int(i / 9)))
            print("--- Finished Pickling ---")
            mol_dfs = []
            del molecules
            gc.collect()



    print("--- Finished ---")


if __name__ == "__main__":
    main()

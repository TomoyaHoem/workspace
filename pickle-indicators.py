import sys
import os
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


def main() -> None:
    print(f"Entry")

    start = time.time()
    # unpickle
    molecules = pd.read_pickle("./pkl/100-fragments.pkl")
    print(molecules.head())
    end = time.time()
    dur = round(end - start, 3)

    print(f"Elapsed time to unpickle: {dur}s")

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

    print(molecules.head())

    # pickle result
    molecules.to_pickle("./pkl/100-fragments-indicators.pkl")
    print("--- Finished Pickling ---")


if __name__ == "__main__":
    main()

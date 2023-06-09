from rdkit import Chem
from rdkit.Chem import AllChem

import sys
import os
import numpy as np
import pandas as pd
import time
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from alive_progress import alive_bar


NUM_MOLS = 2000


def col(val: bool) -> str:
    if val:
        return "red"
    return "blue"


def dominates(a: list, b: list) -> bool:
    dominate = False

    # check QED
    if b[0] > a[0]:
        return False
    if a[0] > b[0]:
        dominate = True

    # check SA
    if b[1] < a[1]:
        return False
    if a[1] < b[1]:
        dominate = True

    return dominate


def findNondominated(mols: pd.DataFrame) -> pd.DataFrame:
    mols["nondominated"] = True

    with alive_bar(NUM_MOLS * NUM_MOLS) as bar:
        for i, row1 in mols.iterrows():
            for j, row2 in mols.iterrows():
                if dominates([row2["QED"], row2["SA"]], [row1["QED"], row1["SA"]]):
                    mols.at[i, "nondominated"] = False
                bar()

    return mols


def main() -> None:
    print(f"Entry")

    start = time.time()
    # unpickle
    molecules = pd.read_pickle("./pkl/100-shards-indicators.pkl")
    end = time.time()
    dur = round(end - start, 3)

    print(f"Elapsed time to unpickle: {dur}s")

    print(molecules.head())

    # plot result
    # Creating figure
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter(molecules["QED"], molecules["LogP"], molecules["SA"], color="green")
    ax.set_xlabel("QED", fontweight="bold")
    ax.set_ylabel("LogP", fontweight="bold")
    ax.set_zlabel("SA", fontweight="bold")
    plt.title("Molecule Indicator")

    # show plot
    # plt.show()

    molecules = molecules.head(NUM_MOLS)

    # find non-dominated molecules
    molecules = findNondominated(molecules)

    # color column depending on domination
    molecules["color"] = molecules["nondominated"].apply(col)

    print(molecules.head())

    # plot pareto front
    molecules.plot.scatter(x="QED", y="SA", c="color")
    plt.show()

    # add fingerprints
    fpGen = AllChem.GetRDKitFPGenerator()

    molecules["Fingerprint"] = molecules["Mol"].apply(fpGen.GetFingerprint)

    print(molecules.head())
    # pickle result
    molecules.to_pickle("./pkl/100-shards-2ksubset-pareto.pkl")
    print("--- Finished Pickling ---")


if __name__ == "__main__":
    main()

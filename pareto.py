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
from colorsys import hsv_to_rgb


NUM_MOLS = 2000


def assignColor(val: int, minval: int, maxval: int) -> list:
    h = (float(val - minval) / (maxval - minval)) * 120

    r, g, b = hsv_to_rgb(h / 360, 1.0, 1.0)
    return r, g, b


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

    # check LogP
    if b[2] > a[2]:
        return False
    if a[2] > b[2]:
        dominate = True

    return dominate


def fastNonDominatedSort(mols: pd.DataFrame) -> pd.DataFrame:
    mols["frontIndex"] = 0

    fronts = []

    if len(mols) > 0:
        fronts.append([])

    dominateMeCount = [0] * len(mols)
    iDominateList = []

    # determine first front
    with alive_bar(NUM_MOLS * NUM_MOLS) as bar:
        for p, row1 in mols.iterrows():
            iDominateList.append([])
            for q, row2 in mols.iterrows():
                if dominates(
                    [row1["QED"], row1["SA"], row1["LogP"]],
                    [row2["QED"], row2["SA"], row2["LogP"]],
                ):
                    iDominateList[p].append(q)
                elif dominates(
                    [row2["QED"], row2["SA"], row2["LogP"]],
                    [row1["QED"], row1["SA"], row1["LogP"]],
                ):
                    dominateMeCount[p] += 1
                bar()
            if dominateMeCount[p] == 0:
                fronts[0].append(p)

    # determine remaining fronts
    frontIndex = 1
    currentFront = fronts[0]
    end = False

    while not end:
        for i, p in enumerate(currentFront):
            for q in iDominateList[p]:
                dominateMeCount[q] -= 1
                if dominateMeCount[q] == 0:
                    if len(fronts) <= frontIndex:
                        fronts.append([])
                    fronts[frontIndex].append(q)
            if i == len(currentFront) - 1:
                if len(iDominateList[p]) == 0:
                    end = True
                    break
                currentFront = fronts[frontIndex]
                frontIndex += 1

    for index in range(len(fronts)):
        for p in fronts[index]:
            mols.at[p, "frontIndex"] = index + 1

    mols.loc[mols["frontIndex"] == 0, "frontIndex"] = len(fronts) + 1

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
    # ax.scatter(molecules["QED"], molecules["LogP"], molecules["SA"], color="green")
    # ax.set_xlabel("QED", fontweight="bold")
    # ax.set_ylabel("LogP", fontweight="bold")
    # ax.set_zlabel("SA", fontweight="bold")
    # plt.title("Molecule Indicator")

    # show plot
    # plt.show()

    molecules = molecules.head(NUM_MOLS)

    # find non-dominated molecules
    molecules = fastNonDominatedSort(molecules)

    # color column depending on domination
    maxval = molecules["frontIndex"].max()
    molecules["color"] = molecules["frontIndex"].apply(assignColor, args=(1, maxval))

    print(molecules.head())

    # # plot pareto front
    for front, m in molecules.groupby("frontIndex"):
        ax.scatter(
            m["QED"], m["LogP"], m["SA"], color=m["color"], label="Front " + str(front)
        )
    ax.set_xlabel("QED", fontweight="bold")
    ax.set_ylabel("LogP", fontweight="bold")
    ax.set_zlabel("SA", fontweight="bold")
    plt.title("Molecule Indicator")
    plt.legend()
    plt.show()

    # # add fingerprints
    # fpGen = AllChem.GetRDKitFPGenerator()

    # molecules["Fingerprint"] = molecules["Mol"].apply(fpGen.GetFingerprint)

    # print(molecules.head())
    # pickle result
    molecules.to_pickle("./pkl/100-shards-2ksubset-pareto.pkl")
    # molecules.to_pickle("./pkl/100-shards-pareto.pkl")
    print("--- Finished Pickling ---")


if __name__ == "__main__":
    main()

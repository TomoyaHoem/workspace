# compare fingerprints with avg / max / min / etc. molecule
# compare different types of fingerprints

# tanimoto

import pandas as pd
import time
import numpy as np
from matplotlib import pyplot as plt

from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs

# 0 3 - 0 6 - 3 6
# 1 4 - 1 7 - 4 7
# 2 5 - 2 8 - 5 8


def plotSim(similarities: pd.DataFrame) -> None:
    fig, axs = plt.subplots(3, 3)
    cols = similarities.columns

    for i in range(3):
        for j in range(3):
            if j > 1:
                k = 3 + i
                l = 6 + i
            else:
                k = i
                l = 3 + i + 3 * j
            axs[i, j].scatter(
                similarities[cols[k]],
                similarities[cols[l]],
                color=similarities["color"],
                s=1.0,
            )
            axs[i, j].set(xlabel=cols[k], ylabel=cols[l])
            axs[i, j].set_xlim(0, 1)
            axs[i, j].set_ylim(0, 1)

    fig.tight_layout(pad=0.5)
    plt.show()


def main() -> None:
    start = time.time()
    # unpickle
    molecules = pd.read_pickle("./pkl/100-shards-2ksubset-pareto.pkl")
    end = time.time()
    dur = round(end - start, 3)

    print(f"Elapsed time to unpickle: {dur}s")
    print(molecules.head())

    maxval = molecules["frontIndex"].max()
    medianval = molecules["frontIndex"].median()

    print(f"Median: {medianval} and Max: {maxval} frontindex.")

    # generate fingerprints

    rdkitfpgen = AllChem.GetRDKitFPGenerator()
    morganfpgen = AllChem.GetMorganGenerator()

    start = time.time()

    molecules["RDKit Fingerprint"] = molecules["Mol"].apply(rdkitfpgen.GetFingerprint)
    molecules["Morgan Fingerprint"] = molecules["Mol"].apply(morganfpgen.GetFingerprint)
    molecules["MACCSKey"] = molecules["Mol"].apply(MACCSkeys.GenMACCSKeys)

    end = time.time()
    dur = round(end - start, 3)

    print(f"Elapsed time to generate fingerprints: {dur}s")

    # get first, middle and last front member reference set
    low = np.random.choice(
        molecules.loc[molecules["frontIndex"] == maxval].index.values.tolist()
    )
    med = np.random.choice(
        molecules.loc[molecules["frontIndex"] == medianval].index.values.tolist()
    )
    hi = np.random.choice(
        molecules.loc[molecules["frontIndex"] == 1].index.values.tolist()
    )

    print(f"low, med and hi indexes {[low, med, hi]}")

    rd_refs = molecules.iloc[[low, med, hi]]["RDKit Fingerprint"].tolist()
    morgan_refs = molecules.iloc[[low, med, hi]]["Morgan Fingerprint"].tolist()
    macc_refs = molecules.iloc[[low, med, hi]]["MACCSKey"].tolist()

    # store similarities in dataframe
    # low -> last front, mid -> middle front, hi -> first front
    similarities = pd.DataFrame()

    # calculate tanimoto similarities between all and reference set
    # rd
    similarities["RD-lo"] = DataStructs.BulkTanimotoSimilarity(
        rd_refs[0], molecules["RDKit Fingerprint"]
    )
    similarities["RD-med"] = DataStructs.BulkTanimotoSimilarity(
        rd_refs[1], molecules["RDKit Fingerprint"]
    )
    similarities["RD-hi"] = DataStructs.BulkTanimotoSimilarity(
        rd_refs[2], molecules["RDKit Fingerprint"]
    )
    # morgan
    similarities["Morg-lo"] = DataStructs.BulkTanimotoSimilarity(
        morgan_refs[0], molecules["Morgan Fingerprint"]
    )
    similarities["Morg-med"] = DataStructs.BulkTanimotoSimilarity(
        morgan_refs[1], molecules["Morgan Fingerprint"]
    )
    similarities["Morg-hi"] = DataStructs.BulkTanimotoSimilarity(
        morgan_refs[2], molecules["Morgan Fingerprint"]
    )
    # macc
    similarities["MACCS-lo"] = DataStructs.BulkTanimotoSimilarity(
        macc_refs[0], molecules["MACCSKey"]
    )
    similarities["MACCS-med"] = DataStructs.BulkTanimotoSimilarity(
        macc_refs[1], molecules["MACCSKey"]
    )
    similarities["MACCS-hi"] = DataStructs.BulkTanimotoSimilarity(
        macc_refs[2], molecules["MACCSKey"]
    )
    similarities["frontIndex"] = molecules["frontIndex"]
    similarities["color"] = molecules["color"]
    similarities.loc[similarities["frontIndex"] != 1, "color"] = "#C2FFC333"

    print(similarities.head())

    no_self_sim = similarities.drop(index=[low, med, hi])

    # plot similarities by contrasting 2 fingerprint methods at a time -> total of 9 (3x3) plots
    plotSim(no_self_sim)


if __name__ == "__main__":
    main()

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def concatPkl() -> None:
    # read pkl files and concatenate to one dataframe
    print("Reading pkls...")
    dfs = []
    for i in range(1, 11):
        dfs.append(pd.read_pickle("./pkl/subset-indicators.pkl_" + str(i)))
    molecules = pd.concat(dfs)
    print("Pkl...")
    molecules.to_pickle("./pkl/subset-indicators.pkl")
    print("Finished pkl!")


def main() -> None:
    print("Entry")

    # concatPkl()
    molecules = pd.read_pickle("./pkl/subset-indicators.pkl")
    print(molecules.info())
    print(molecules.head())

    # plot histograms of columns
    hist = molecules.hist()
    plt.show(block=False)

    # compare to sampled hitograms
    mol_sample = molecules.sample(100)
    hist_s = mol_sample.hist()
    plt.show()


if __name__ == "__main__":
    main()

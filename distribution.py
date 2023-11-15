import os
import sys

import pandas as pd


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

    concatPkl()

    # plot histograms of columns

    # compare to sampled hitograms


if __name__ == "__main__":
    main()

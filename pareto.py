from rdkit import Chem
import os
import numpy as np
import pandas as pd


def main() -> None:
    print(f"Entry")

    # unpickle
    molecules = pd.read_pickle("./pkl/shards.pkl")
    print(molecules.head())

    # evaluate all molecules for logp, sa, qed

    # plot result

    # apply non-dominated sorting

    # plot pareto front


if __name__ == "__main__":
    main()

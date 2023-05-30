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
    molecules = pd.read_pickle("./pkl/100-shards-indicators.pkl")
    end = time.time()
    dur = round(end - start, 3)

    print(f"Elapsed time to unpickle: {dur}s")

    print(molecules.head())

    # plot result
    molecules.plot(kind="scatter", x="QED", y="LogP")
    plt.show()

    # apply non-dominated sorting

    # plot pareto front


if __name__ == "__main__":
    main()

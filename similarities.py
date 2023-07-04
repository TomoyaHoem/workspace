# compare fingerprints with avg / max / min / etc. molecule
# compare different types of fingerprints

# compare different types of similarities: tanimoto

import pandas as pd
import time


def main() -> None:
    start = time.time()
    # unpickle
    molecules = pd.read_pickle("./pkl/100-shards-2ksubset-pareto.pkl")
    end = time.time()
    dur = round(end - start, 3)

    print(f"Elapsed time to unpickle: {dur}s")
    print(molecules.head())


if __name__ == "__main__":
    main()

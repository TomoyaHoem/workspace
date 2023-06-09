# 1. Random one-point crossover (or n-point), repeat until valid found

# 2. Truncate randomly from left and right, concatenate truncations, fix parantheses, repeat until valid

# 3. Consider rings, non-rings

# 4. Include valence rules

# 5. Predefine collection of constraints
import time
import pandas as pd


def main() -> None:
    start = time.time()
    # unpickle
    molecules = pd.read_pickle("./pkl/100-shards-2ksubset-pareto.pkl")
    end = time.time()
    dur = round(end - start, 3)

    print(f"Elapsed time to unpickle: {dur}s")
    print(molecules.head())

    # get a sample of smiles

    # try crossover variants


if __name__ == "__main__":
    main()

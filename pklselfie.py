import time
import pandas as pd
import selfies as sf

def main():
    print("Entry")
    mols = pd.read_pickle("./pkl/subset-indicators.pkl")
    start = time.time()
    mols["SELFIES"] = mols["Smiles"].apply(sf.encoder)
    end = time.time()
    dur = round(end - start, 3)
    print(f"Elapsed time to encode SELFIES: {dur}s")
    print(mols.head())
    print(mols.info())
    mols.to_pickle("./pkl/subset-selfies.pkl")

if __name__ == "__main__":
    main()
import time
import numpy as np
import pandas as pd
from indices.baroni_urbani_buser import BaroniUrbaniBuser
from rdkit import Chem
from rdkit.Chem import AllChem


def fingerprints(molecules):
    start = time.time()
    mol_fps = []
    for molecule in molecules:
        mol_fps.append(
            np.array(
                AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles(molecule), 4, nBits=1024
                )
            )
        )
    end = time.time()
    dur = round(end - start, 3)

    # print(f"Elapsed time to calculate fingerprints: {dur}s")
    return np.array(mol_fps)


def internal_similarity(molecules, c_threshold=None, w_factor="fraction") -> float:
    """
    Returns the extended BuB similarity for the list of molecules
    """
    fps = fingerprints(molecules)
    start = time.time()
    multiple_comparisons_index = BaroniUrbaniBuser(
        fingerprints=fps, c_threshold=c_threshold, w_factor=w_factor
    )
    end = time.time()
    dur = round(end - start, 3)

    # print(f"Elapsed time to calculate similarity: {dur}s")
    return multiple_comparisons_index.__dict__["BUB_1sim_dis"]


def main() -> None:
    molecules = pd.read_pickle("./pkl/subset-indicators.pkl")
    print(molecules.info())
    print(molecules.head())

    num = 500_000
    sample = molecules.sample(num)

    mol_list = sample["Smiles"].tolist()

    start = time.time()
    print(internal_similarity(mol_list))
    end = time.time()
    dur = round(end - start, 3)

    print(
        f"Elapsed time to calculate fingerprints and similarity for {num} molecules: {dur}s"
    )


if __name__ == "__main__":
    main()

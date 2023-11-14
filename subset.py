import os
import time
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

allowed = {"C", "N", "S", "O", "F", "Cl", "Br", "H"}

_mcf = pd.read_csv(os.path.join(os.getcwd(), "filters", "mcf.csv"))
_pains = pd.read_csv(
    os.path.join(os.getcwd(), "filters", "wehi_pains.csv"), names=["smarts", "names"]
)
_filters = [
    Chem.MolFromSmarts(x) for x in pd.concat([_mcf, _pains], sort=True)["smarts"].values
]


def filter_mol(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    # 1. SMILES valid
    if mol is None:
        return False
    # 2. less than 7 RTB
    if Descriptors.NumRotatableBonds(mol) > 7:
        return False
    # 3. less than 8 cycles
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 0 and any(len(x) >= 8 for x in ring_info.AtomRings()):
        return False
    # 4. Charged / Allowed
    h_mol = Chem.AddHs(mol)
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return False
    if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):
        return False
    # 5. Filters
    if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
        return False
    return True


def main() -> None:
    print(f"Reading Data...")

    start = time.time()

    molecules = pd.read_pickle("./pkl/subset-lfs.pkl")

    end = time.time()
    dur = round(end - start, 3)
    print(f"Elapsed time to unpickle: {dur}s")

    print(len(molecules))
    print(molecules.head())

    start = time.time()
    molecules = molecules[molecules["Smiles"].apply(filter_mol)]
    end = time.time()
    dur = round(end - start, 3)
    print(f"Elapsed time to apply filters: {dur}s")

    print(len(molecules))
    print(molecules.head())

        # pickle dataframe
    print("pickle...")
    molecules.to_pickle("./pkl/filtred-subset-lfs.pkl")
    print("--- Finished Pickling ---")


if __name__ == "__main__":
    main()

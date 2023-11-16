import numpy as np
from indices.baroni_urbani_buser import BaroniUrbaniBuser
from rdkit import Chem
from rdkit.Chem import AllChem


def fingerprints(molecules):
    mol_fps = []
    for molecule in molecules:
        mol_fps.append(
            np.array(
                AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles(molecule), 4, nBits=1024
                )
            )
        )
    return np.array(mol_fps)


def internal_similarity(molecules, c_threshold=None, w_factor="fraction") -> float:
    """
    Returns the extended BuB similarity for the list of molecules
    """
    fps = fingerprints(molecules)
    multiple_comparisons_index = BaroniUrbaniBuser(
        fingerprints=fps, c_threshold=c_threshold, w_factor=w_factor
    )
    return multiple_comparisons_index.__dict__["BUB_1sim_wdis"]


def main() -> None:
    a = [
        "OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N",
        "CC(C)[C@@]12C[C@@H]1[C@@H](C)C(=O)C2",
        "CC[C@H](O1)CC[C@@]12CCCO2",
        "CC(=O)OCCC(/C)=C\C[C@H](C(C)=C)CCC=C",
        "O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5",
        "CCC[C@@H](O)CC/C=C/C=C/C#CC#C/C=C/CO",
        "CN1CCC[C@H]1c2cccnc2",
        "CCc(c1)ccc2[n+]1ccc3c2[nH]c4c3cccc4",
        "O=Cc1ccc(O)c(OC)c1",
        "O=Cc1ccc(O)c(OC)c1",
        "COc1cc(C=O)ccc1O",
    ]

    print(internal_similarity(a))


if __name__ == "__main__":
    main()

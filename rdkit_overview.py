from rdkit import Chem
from rdkit.Chem import Draw

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    m = Chem.MolFromSmiles("C=CCc1ccc(OCC(=O)N(CC)CC)c(OC)c1")
    if m is None:
        print("Invalid Molecule")
        return

    img = Draw.MolToImage(m)
    img.show()


if __name__ == "__main__":
    main()

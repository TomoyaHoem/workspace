from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import QED

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    m = Chem.MolFromSmiles("C=CCc1ccc(OCC(=O)N(CC)CC)c(OC)c1")
    if m is None:
        print("Invalid Molecule")
        return

    m2 = Chem.MolFromSmiles("Cc1ccccc1")

    img = Draw.MolToImage(m)
    img.show()

    print(QED.default(m))
    print(QED.properties(m))

    print(QED.default(m2))
    print(QED.properties(m2))


if __name__ == "__main__":
    main()

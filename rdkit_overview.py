from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem import RDConfig
import os
import sys

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
# now you can import sascore!
import sascorer  # type: ignore

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

    logP = Crippen.MolLogP(m)
    print(f"Molecule's LogP value is: {logP}")

    s = sascorer.calculateScore(m)
    print(f"Molecule's SAscore: {s}")

    print(QED.default(m))
    print(QED.properties(m))

    print(QED.default(m2))
    print(QED.properties(m2))


if __name__ == "__main__":
    main()

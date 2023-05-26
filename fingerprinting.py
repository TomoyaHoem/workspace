from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def main() -> None:
    m = Chem.MolFromSmiles("CCC")

    fpGen = AllChem.GetRDKitFPGenerator()
    fp = fpGen.GetFingerprint(m)

    print(np.nonzero(np.array(fp)))
    print(fp.ToBitString())


if __name__ == "__main__":
    main()

from rdkit import Chem
from rdkit.Chem import Draw
import os


def main() -> None:
    cwd = os.getcwd()
    path = cwd + "\ZINCSMILES\AA\AAAA.smi"

    suppl = Chem.SmilesMolSupplier(path)

    for mol in suppl:
        img = Draw.MolToImage(mol)
        img.show()
        input("Press any key to show next molecule...")


if __name__ == "__main__":
    main()

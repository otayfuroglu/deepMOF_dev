#
from openbabel import openbabel
import os


def loadMolWithOB(mol_path):

    _, fl_format = os.path.splitext(mol_path)
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat(fl_format)
    ob_mol = openbabel.OBMol()
    obConversion.ReadFile(ob_mol, mol_path)

    return ob_mol


def getRmsdOBMol(mol1, mol2):

    obAlign = openbabel.OBAlign(mol1, mol2)
    obAlign.Align()
    return obAlign.GetRMSD()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-ref", type=str, required=True)
    parser.add_argument("-mol", type=str, required=True)
    args = parser.parse_args()

    mol1 = loadMolWithOB(args.ref)
    mol2 = loadMolWithOB(args.ref)
    print(getRmsdOBMol(mol1, mol2))

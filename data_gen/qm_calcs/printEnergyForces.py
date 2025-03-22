#
from ase.io import read
import argparse

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-in_extxyz", type=str, required=True)
args = parser.parse_args()


in_extxyz = args.in_extxyz

with open(f"{in_extxyz.split('/')[-1].split('.')[0]}_energy.csv", "w") as fl:
    print("FileNames,Energy (eV/atom)", file=fl)
    for atoms in read(in_extxyz, index=":"):
        print(f"{atoms.info['label']},{atoms.get_potential_energy()/len(atoms)}", file=fl)


with open(f"{in_extxyz.split('/')[-1].split('.')[0]}_forces.csv", "w") as fl:
    print("FileNames,Forces (eV/A)", file=fl)
    for atoms in read(in_extxyz, index=":"):
        for force in atoms.get_forces().flatten():
            print(f"{atoms.info['label']},{force}", file=fl)



from ase import Atoms
from ase.io import write, read
from ase.calculators.singlepoint import SinglePointCalculator
import argparse

import json



def getSelfEnergy(atoms):
    total_self_energy = 0
    for symbol in symbols_selfenergy_pair.keys():
        num_atoms = len([atom for atom in atoms if atom.symbol == symbol])
        self_energy = num_atoms * float(symbols_selfenergy_pair[symbol])
        total_self_energy += self_energy
    return total_self_energy

# ORCA 6.0  RPBE D3BJ DEF2-TZVP
#  atomic_energies={"C": "-1028.3626838002976",
#                   "O": "-2040.4160983992801",
#                   "H": "-13.737306152573883",
#                   "Mg": "-5443.727367053916",
#                  }
#
#  symbols = ['C', 'O', 'H', 'Mg']
#



parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-extxyz_path", type=str, required=True, help="..")
parser.add_argument("-symbols_selfenergy_pair", type=str, required=True, help="")
args = parser.parse_args()

extxyz_path = args.extxyz_path
symbols_selfenergy_pair = json.loads(args.symbols_selfenergy_pair)

atoms_list = read(extxyz_path, index=":")
for atoms in atoms_list:
    total_energy = atoms.get_potential_energy()
    self_energy = getSelfEnergy(atoms)
    cohesive_energy = total_energy - self_energy
    atoms.calc = SinglePointCalculator(atoms, energy=cohesive_energy, forces=atoms.get_forces())
    atoms.get_potential_energy()
    write(f"cohesive_{extxyz_path.split('/')[-1]}", atoms, append=True)


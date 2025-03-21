from ase import Atoms
from ase.io import write, read
from ase.calculators.singlepoint import SinglePointCalculator
import argparse

import json



def substractSelfEnergy(atoms):
    total_self_energy = 0
    for symbol in symbols_selfenergy_pair.keys():
        num_atoms = len([atom for atom in atoms if atom.symbol == symbol])
        self_energy = num_atoms * float(symbols_selfenergy_pair[symbol])
        total_self_energy += self_energy
    return total_self_energy

#  atomic_energies={"C": "-1025.34896263954",
#                   "O": "-2034.97553102427",
#                   "H": "-13.5576149854593",
#                   "Mg": "-1479.90402789208",
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
    self_energy = substractSelfEnergy(atoms)
    cohesive_energy = total_energy-self_energy
    atoms.calc = SinglePointCalculator(atoms, energy=cohesive_energy, forces=atoms.get_forces())
    atoms.get_potential_energy()
    write(f"cohesive_{extxyz_path.split('/')[-1]}", atoms, append=True)


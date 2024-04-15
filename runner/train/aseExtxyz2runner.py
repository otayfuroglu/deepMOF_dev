#
from ase.io import read
from ase import units
from ase.db import connect
import numpy as np

import tqdm
import argparse
import os
#  import shutil


u = units.create_units("2014")
EV2HARTREE = 1.0 / u["Hartree"]
ANG2BOHR = 1.0 / u["Bohr"]
EV2HARTREE = 1.0
ANG2BOHR = 1.0

def extxyz2Runner(atoms_list):
    fl = open(f"input.data", "w")
    for i, atoms in enumerate(tqdm.tqdm(atoms_list)):
        atoms_prep_list = [["begin"]]
        if any(atoms.pbc):
            atoms_prep_list += [["lattice    {:15.8f}    {:15.8f}    {:15.8f}".format(
                    cell[0], cell[1], cell[2])] for cell in atoms.cell * ANG2BOHR]

        atom_template = 'atom {:15.8f} {:15.8f} {:15.8f} {:2s} {:15.8f} {:15.8f} {:15.8f} {:15.8f} {:15.8f}'
        atoms_prep_list += [[atom_template.format(
            position[0], position[1], position[2],
            symbol, charge, 0.0,
            forces[0], forces[1], forces[2])]
            for position, symbol, charge, forces in zip(
                (atoms.positions * ANG2BOHR).tolist(),
                atoms.symbols,
                atoms.get_initial_charges(),
                (np.array(atoms.get_forces()) * (EV2HARTREE/ANG2BOHR)).tolist())]

        atoms_prep_list += [["energy ",
                             atoms.get_potential_energy() * EV2HARTREE], ["charge 0.0"], ["end"]]

        for line in atoms_prep_list:
            for item in line:
                fl.write(str(item))
            fl.write("\n")
    fl.close()

def rndExtxzy2Runner(atoms_list):
    import random
    rand_list = random.sample(range(db.count()), N)
    db = db.select()
    fl = open(f"input.data.rand{N}", "w")
    for i, atoms in enumerate(tqdm.tqdm(atoms_list)):
        if i in rand_list:
            # remove random value for efficiency
            rand_list.remove(i)
            atoms_prep_list = [["begin"]]
            atoms_prep_list += [["lattice    {:15.8f}    {:15.8f}    {:15.8f}".format(
                    cell[0], cell[1], cell[2])] for cell in atoms.cell * ANG2BOHR]

            atom_template = 'atom {:15.8f} {:15.8f} {:15.8f} {:2s} {:15.8f} {:15.8f} {:15.8f} {:15.8f} {:15.8f}'
            atoms_prep_list += [[atom_template.format(
                position[0], position[1], position[2],
                symbol, charge, 0.0,
                forces[0], forces[1], forces[2])]
                for position, symbol, charge, forces in zip(
                    (atoms.positions * ANG2BOHR).tolist(),
                    atoms.symbols,
                    atoms.get_initial_charges(),
                    (np.array(atoms.get_forces()) * (EV2HARTREE/ANG2BOHR)).tolist())]

            atoms_prep_list += [["energy ",
                                 atoms.get_potential_energy() * EV2HARTREE], ["charge 0.0"], ["end"]]

            for line in atoms_prep_list:
                for item in line:
                    fl.write(str(item))
                fl.write("\n")
    fl.close()

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-extxyz_path", type=str, required=True)
parser.add_argument("-N", type=int, default=0, required=False)
args = parser.parse_args()


atoms_list = read(args.extxyz_path, index=":")
N = args.N

if N == 0:
    extxyz2Runner(atoms_list)
else:
    rndExtxzy2Runner(atoms_list)


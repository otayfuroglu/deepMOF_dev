
#  from ase.io.lammpsrun import read_lammps_dump_text
from ase.io import read
from ase import units
import numpy as np

import tqdm
import argparse
import os
#  import shutil


u = units.create_units("2014")
EV2HARTREE = 1.0 / u["Hartree"]
ANG2BOHR = 1.0 / u["Bohr"]

def aseDb2Runner(traj, atomic_numbers):
    #  traj = traj.select()
    fl = open(f"input.data", "w")
    for i, row in enumerate(tqdm.tqdm(traj)):
        # flammps tpye to atom nummer
        row.set_atomic_numbers(atomic_numbers)
        # remove random value for efficiency
        atoms_prep_list = [["begin"]]
        atoms_prep_list += [["lattice    {:15.8f}    {:15.8f}    {:15.8f}".format(
                cell[0], cell[1], cell[2])] for cell in row.cell * ANG2BOHR]

        atom_template = 'atom {:15.8f} {:15.8f} {:15.8f} {:2s} {:15.8f} {:15.8f} {:15.8f} {:15.8f} {:15.8f}'
        atoms_prep_list += [[atom_template.format(
            position[0], position[1], position[2],
            symbol, 0.0, 0.0, 0.0, 0.0, 0.0)] # symbol, charge, 0.0, forces
            for position, symbol, in zip( row.positions * ANG2BOHR, row.symbols)]

        atoms_prep_list += [["energy ", 0.], ["charge 0.0"], ["end"]]

        for line in atoms_prep_list:
            for item in line:
                fl.write(str(item))
            fl.write("\n")
    fl.close()

def rndAseDb2Runner(traj, atomic_numbers):
    import random
    rand_list = random.sample(range(len(traj)), N)
    #  traj = traj.select()
    fl = open(f"input.data.rand{N}", "w")
    for i, row in enumerate(tqdm.tqdm(traj)):

        # flammps tpye to atom nummer
        row.set_atomic_numbers(atomic_numbers)

        if i in rand_list:
            # remove random value for efficiency
            rand_list.remove(i)
            atoms_prep_list = [["begin"]]
            atoms_prep_list += [["lattice    {:15.8f}    {:15.8f}    {:15.8f}".format(
                    cell[0], cell[1], cell[2])] for cell in row.cell * ANG2BOHR]

            atom_template = 'atom {:15.8f} {:15.8f} {:15.8f} {:2s} {:15.8f} {:15.8f} {:15.8f} {:15.8f} {:15.8f}'
            atoms_prep_list += [[atom_template.format(
                position[0], position[1], position[2],
                symbol, 0.0, 0.0, 0.0, 0.0, 0.0)] # symbol, charge, 0.0, forces
                for position, symbol, in zip( row.positions * ANG2BOHR, row.symbols)]

            atoms_prep_list += [["energy ", 0.], ["charge 0.0"], ["end"]]

            for line in atoms_prep_list:
                for item in line:
                    fl.write(str(item))
                fl.write("\n")
    fl.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Give something ...")
    parser.add_argument("-traj", type=str, required=True)
    parser.add_argument("-N", type=int, default=0, required=False)
    args = parser.parse_args()

    atom_type_number_pair = {1:13, 2:3, 3:1} # Al, Li, H
    traj = read(args.traj, format="lammps-dump-text", index=":")
    atomic_numbers = [atom_type_number_pair[key] for key in traj[0].get_atomic_numbers()]

    N = args.N
    if N == 0:
        aseDb2Runner(traj, atomic_numbers)
    else:
        rndAseDb2Runner(traj, atomic_numbers)

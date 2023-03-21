
#  from ase.io.lammpsrun import read_lammps_dump_text
from ase.io import read
from ase.io.extxyz import write_extxyz
from ase import units
import numpy as np

import tqdm
import argparse
import os
#  import shutil


u = units.create_units("2014")
EV2HARTREE = 1.0 / u["Hartree"]
ANG2BOHR = 1.0 / u["Bohr"]

def lammps2atoms(lammps_atoms, atomic_numbers):

        # lammps tpye to atom nummer
     lammps_atoms.set_atomic_numbers(atomic_numbers)
     return lammps_atoms

def rndLammps2AseDB(fl_path, traj, atomic_numbers):
    import random
    rand_list = random.sample(range(len(traj)), N)
    for i, lammps_atoms in enumerate(tqdm.tqdm(traj)):
        if i in rand_list:
            # remove random value for efficiency
            rand_list.remove(i)
            atoms = lammps2atoms(lammps_atoms, atomic_numbers)
            write_extxyz(fl_path, atoms, append=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Give something ...")
    parser.add_argument("-traj", type=str, required=True)
    parser.add_argument("-skip", type=int, default=0, required=False)
    parser.add_argument("-stepsize", type=int, default=1, required=False)
    parser.add_argument("-N", type=int, default=0, required=False)
    args = parser.parse_args()

    atom_type_number_pair = {1:13, 2:3, 3:1} # Al, Li, H
    index = slice(args.skip, -1, args.stepsize) # sliced index for lammps traj
    traj = read(args.traj, format="lammps-dump-text", index=index)
    atomic_numbers = [atom_type_number_pair[key] for key in traj[0].get_atomic_numbers()]

    N = args.N
    stepsize = args.stepsize
    if N == 0:
        for i, lammps_atoms in enumerate(tqdm.tqdm(traj)):
            # flammps tpye to atom nummer
            atoms = lammps2atoms(lammps_atoms, atomic_numbers)
            write_extxyz("input.ext.xyz", atoms, append=True)
    else:
        fl_path = f"input.ext.xyz.rand{N}"
        rndLammps2AseDB(fl_path, traj, atomic_numbers)

#! /truba/home/yzorlu/miniconda3/bin/python

from ase.io import read, write
from ase.visualize import view
from ase.build import bulk
import numpy as np

from multiprocessing import Pool
from itertools import product
import os
import tqdm
import argparse


"""ase Atoms object; rattle
Randomly displace atoms.
This method adds random displacements to the atomic positions,
taking a possible constraint into account. The random numbers are drawn
from a normal distribution of standard deviation stdev.
For a parallel calculation, it is important to use the same seed on all processors!
"""


def scale_atoms_distence(atoms, scale_factor):
    #  atoms = atoms.copy()
    atoms.center(vacuum=0.0)
    atoms.set_cell(scale_factor * atoms.cell, scale_atoms=True)
    #print(atoms.cell)
    #write("test.xyz", atoms)
    #view(atoms)
    return atoms

#atoms = read("./fragments/mof5_f1.xyz")
#scale_atoms_distence(atoms, 2)


def random_scale_direction(direction):
    return np.random.uniform(0.96 * direction, 1.10 * direction)

def calc_displacement_atom(directions_distance):
    return np.sqrt(sum(direction**2 for direction in directions_distance))


def displaced_atomic_positions(atom_positions):

    while True:
        n_atom_positions = np.array([random_scale_direction(direction)
                                    for direction in atom_positions])
        if calc_displacement_atom(atom_positions - n_atom_positions) <= 0.16:
            return n_atom_positions

def get_non_equ_geom(flpath):

    #  NON_EQU_XYZ_DIR = BASE_DIR = "non_equ_geoms"
    #  if not os.path.exists(NON_EQU_XYZ_DIR):
    #      os.mkdir(NON_EQU_XYZ_DIR)

    file_name= flpath.split("/")[-1]
    file_base = file_name.replace(".xyz", "")


    scale_range = (0.96, 1.11)
    scale_step = 0.00010

    # scale atomic positions
    for i, scale_factor in enumerate(np.arange(scale_range[0], scale_range[1], scale_step)):
        # reread every scaling iteration for escape cumulative scaling
        atoms = read(flpath)
        atoms = scale_atoms_distence(atoms, scale_factor)

        # randomlu displace atomic positions
        for atom in atoms:
            atom.position = displaced_atomic_positions(atom.position)

        #  write("{}/{}_".format(NON_EQU_XYZ_DIR, file_base)+"{0:0>5}".format(i)+".xyz", atoms)
        atoms.info["label"] = file_base + "_" + "{0:0>5}".format(i)
        write(f"non_equ_geoms_{file_base}.extxyz", atoms, append=True)

def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-flpath", type=str, required=True, help="give hdf5 file base")
    args = parser.parse_args()
    flpath = args.flpath
    get_non_equ_geom(flpath)


main()


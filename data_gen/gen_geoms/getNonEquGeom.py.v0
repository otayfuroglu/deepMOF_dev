#! /truba/home/yzorlu/miniconda3/bin/python

from ase.io import read, write
from ase.visualize import view
from ase.build import bulk
import numpy as np

from multiprocessing import Pool
from itertools import product
import os
import tqdm

"""ase Atoms object; rattle
Randomly displace atoms.
This method adds random displacements to the atomic positions,
taking a possible constraint into account. The random numbers are drawn
from a normal distribution of standard deviation stdev.
For a parallel calculation, it is important to use the same seed on all processors!
"""

mof_num = 10
frag_num = 6
BASE_DIR = "/truba_scratch/yzorlu/deepMOF/HDNNP"
XYZ_DIR = BASE_DIR + "/prepare_data/geomFiles/IRMOFSeries/fragments_%s" %frag_num
NON_EQU_XYZ_DIR = BASE_DIR + "/prepare_data/outOfSFGeomsIRMOFs%s" %mof_num

if not os.path.exists(NON_EQU_XYZ_DIR):
    os.mkdir(NON_EQU_XYZ_DIR)

#file_bases = ["mof5_new_f2", "mof5_new_f3", "mof5_new_f4", "mof5_new_f5", "mof5_new_f6"]
#file_bases = ["mof5_new_single", ]

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

def get_non_equ_geom(file_name):

    file_base = file_name.replace(".xyz", "")


    scale_range = (0.96, 1.11)
    scale_step = 0.00007

    # scale atomic positions
    for i, scale_factor in enumerate(np.arange(scale_range[0], scale_range[1], scale_step)):
        # reread every scaling iteration for escape cumulative scaling
        atoms = read("%s/%s.xyz" %(XYZ_DIR, file_base))
        atoms = scale_atoms_distence(atoms, scale_factor)

        # randomlu displace atomic positions
        for atom in atoms:
            atom.position = displaced_atomic_positions(atom.position)

        write("{}/{}_".format(NON_EQU_XYZ_DIR, file_base)+"{0:0>5}".format(i)+".xyz", atoms)

def main(n_proc):

    #  file_names = [file_name for file_name in os.listdir(XYZ_DIR) if ".xyz" in file_name]
    file_names = ["irmofseries%s_f%s.xyz" %(mof_num, frag_num)]

    pool = Pool(processes=n_proc)

    result_list_tqdm = []

    # implementation of  multiprocessor in tqdm. Ref.https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
    for result in tqdm.tqdm(pool.imap_unordered(func=get_non_equ_geom, iterable=file_names), total=len(file_names)):
        result_list_tqdm.append(result)

main(80)


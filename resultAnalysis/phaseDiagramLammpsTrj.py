#
from ase import Atoms
from ase.io import read, write
from ase.db import connect

from ase.geometry import cell_to_cellpar, cellpar_to_cell
from coordNum import coordnum
import os
import tqdm
import numpy as np

#  from numba import jit, float32

import itertools
import argparse
import shutil
#


def atoms2Ascii(atoms):

    # number of atoms
    atoms_prep_list = [[f"{len(atoms):7.0f}"]]

    # cell parameters
    cell = cellpar_to_cell(cell_to_cellpar(atoms.cell))
    dxx = cell[0, 0]
    dyx, dyy = cell[1, 0:2]
    dzx, dzy, dzz = cell[2, 0:3]

    cell_template = '{:15.12f} {:15.12f} {:15.12f}'
    atoms_prep_list += [cell_template.format(dxx, dyx, dyy)]
    atoms_prep_list += [cell_template.format(dzx, dzy, dzz)]

    # positons and symbols
    atom_template = '{:15.12f} {:15.12f} {:15.12f} {:2s}'
    atoms_prep_list += [[atom_template.format(position[0], position[1], position[2], symbol)]
            for position, symbol, in zip( (atoms.positions).tolist(), atoms.symbols)]

    with open(f"tmp.ascii", "w") as fl:
        for line in atoms_prep_list:
            for item in line:
                fl.write(str(item))
            fl.write("\n")


def lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair):
    symbols = [atom_type_symbol_pair[key] for key in lammps_atoms.get_atomic_numbers()]
    return Atoms(symbols=symbols, positions=lammps_atoms.positions, cell=lammps_atoms.cell)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Give something ...")
    #  parser.add_argument("-trj_path", type=str, required=True, help="..")
    parser.add_argument("-basedir", type=str, required=True, help="..")
    parser.add_argument("-lastframes", type=int, required=True, help="..")
    args = parser.parse_args()
    basedir = args.basedir

    idxes = slice(args.lastframes, -1, 1)
    atom_type_symbol_pair = {1:"Al", 2:"Li", 3:"H"}

    fl = open(f"{basedir}/data.dat", "w")
    fl.write("Temp,Press,AvVol,AvMeanCoord,AvMaxCoord,AvMinCoord,AvStdCoord,\n")

    #  dirs = sorted([dir_ for dir_ in os.listdir(basedir) if "Bar" in dir_])
    #  dirs = sorted([dir_ for dir_ in dirs if not "." in dir_])
    #  if len(dirs) == 0:

    #  dirs = file_names
    #  for dir_ in dirs:
    #      split_dir = dir_.split("_")
    #      press = int(split_dir[0].replace("Bar", ""))
    #      temp = int(split_dir[1].replace("K", ""))
        #  print(press, temp)
    file_names = sorted([dir_ for dir_ in os.listdir(basedir) if ".lammpstrj" in dir_])
    for file_name in file_names:
        file_base = file_name.replace(".lammpstrj", "")
        #  print(file_base)
        split_file_base = file_base.split("_")
        press = int(split_file_base[1].replace("Bar", ""))
        temp = int(split_file_base[2].replace("K", ""))


        #  lammps_trj_path = f"{basedir}/{dir_}/alanates_{dir_}.lammpstrj"
        lammps_trj_path = f"{basedir}/{file_name}"
        try:
            lammps_trj = read(lammps_trj_path, format="lammps-dump-text", index=idxes, parallel=True)
        except:
            print(file_base)
            continue
        len_trj = len(lammps_trj)

        pot_es = []
        vols = []
        mean_coord_nums = []
        max_coord_nums = []
        min_coord_nums = []
        std_coord_nums = []
        for i, lammps_atoms in enumerate(lammps_trj):
            atoms = lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair)
            #  pot_es += [atoms.get_potential_energy()]
            vols += [atoms.get_volume()]

            atoms2Ascii(atoms)
            nat = len(atoms)
            coordnum("tmp.ascii", "tmp.extxyz", nat, atom_symb="Al")
            atoms = read("tmp.extxyz")
            coordnums = atoms.arrays["coordn"]
            elem_coordnums = coordnums[coordnums > 0.0]
            max_coordnum = elem_coordnums.max()
            min_coordnum = elem_coordnums.min()
            std_coordnum = elem_coordnums.std()
            mean_coordnum = elem_coordnums.mean()
            os.remove("tmp.ascii")
            os.remove("tmp.extxyz")

            mean_coord_nums += [mean_coordnum]
            max_coord_nums += [max_coordnum]
            min_coord_nums += [min_coordnum]
            std_coord_nums += [std_coordnum]


        av_vol = sum(vols) / len_trj
        av_mean_coordnum = sum(mean_coord_nums) / len_trj
        av_max_coordnum = sum(max_coord_nums) / len_trj
        av_min_coordnum = sum(min_coord_nums) / len_trj
        av_std_coordnum = sum(std_coord_nums) / len_trj

        fl.write(f"{temp},{press},{av_vol},{av_mean_coordnum},{av_max_coordnum},{av_min_coordnum},{av_std_coordnum}\n")
        fl.flush()
        #  break

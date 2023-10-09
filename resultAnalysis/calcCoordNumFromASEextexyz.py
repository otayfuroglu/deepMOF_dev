#
from ase import Atoms
from ase.io import read, write
from ase.db import connect
from ase.neighborlist import (NeighborList,
                              NewPrimitiveNeighborList,
                              PrimitiveNeighborList,
                              natural_cutoffs)
from pymatgen.core import Structure
from pymatgen.analysis.local_env import  (BrunnerNN_reciprocal,
                                          BrunnerNN_real,
                                          CrystalNN,
                                          MinimumDistanceNN)

from pathlib import Path
import os
import tqdm
import numpy as np

#  from numba import jit, float32

from multiprocessing import Pool
import itertools
import argparse
#


#  @jit(float32(float32, float32, float32))
def f_cut(d_kl, Tx, Vx):
    #  print(d_kl, Tx, Vx)

    K = (d_kl - Tx) / (Vx - Tx)
    f = np.array((2 * K + 1) * ((K - 1) ** 2))
    f[d_kl<Tx] = 1
    f[Vx<d_kl] = 0
    #  print(f)
    return f


def get_1NN2NN_distances(nn, struc, center_atom_i):
    nn_distances = []
    site_idexes = []
    for shel_i in [1, 2]:
        distances = []
        for site in nn.get_nn_shell_info(struc, center_atom_i, shel_i):
            #  print(site)
            site_i = site["site_index"]
            if not site_i in site_idexes:
                site_idexes.append(site_i)
                distances += [struc.get_distance(center_atom_i, site_i)]
        #  if len(distances) == 0:
            #  distances.append(0)
        try:
            nn_distances.append(min(distances))
        except:
            return 0, 0.0
    return site_idexes, nn_distances


def get_coord_num(nn, atoms, center_atom_i, replica):

    fl_name = "tmp"
    write(f"{fl_name}.cif", atoms)
    struc = Structure.from_file(f"{fl_name}.cif") #read in CIF as Pymatgen Structure
    if replica > 1:
        struc.make_supercell([replica, replica, replica])

    coord_num = 0
    site_idexes, nn_distances = get_1NN2NN_distances(nn, struc, center_atom_i)
    if nn_distances == 0.0:
        return None

    passed_site_idexes = []# to get under control duplication and self correlation
    for shel_i in [1, 2]:
        for site_i in site_idexes:
            #  print(site_i)
            if not site_i in passed_site_idexes: # to get under control duplication and self correlation
                passed_site_idexes.append(site_i)
                distance = struc.get_distance(center_atom_i, site_i)
                coord_num += f_cut(d_kl=distance, Tx=nn_distances[0], Vx=nn_distances[1])
    return coord_num


def get_coord_numByWeigth(struc, center_atom_i):
    coord_num = 0
    for site in nn.get_nn_info(struc, center_atom_i):
            coord_num += site["weight"]
    return coord_num


def get_coord_numAse(center_atom_i):
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs=cutoffs)
    nl.update(atoms)
    indices, offsets = nl.get_neighbors(center_atom_i)
    for i, offset in zip(indices, offsets):
        #  print(i)
        print(atoms.positions[i] + np.dot(offset, atoms.get_cell()))
    pass


def task(idx):
    atoms = atoms_list[idx]
    Al_index = [atom.index for atom in atoms if atom.symbol == "Al"]
    center_atom_i = Al_index[0]


    coord_num = get_coord_num(nn, atoms, center_atom_i, replica=1)
    if coord_num is None:
        return None
    elif coord_num <=1:
        coord_num = get_coord_num(nn, atoms, center_atom_i, replica=2)
    return f"frame_{idx*args.interval}", coord_num
    #  print(coord_num)

    #  if coord_num == 1:
    #      center_atom_i = Al_index[1]
    #      coord_num = get_coord_num()
    #      #  print("other index", get_coord_num())
    #      if coord_num == 1:
    #          struc.make_supercell([2, 2, 2])
    #          coord_num = get_coord_num()
    #          if coord_num == 1:
    #              continue
    #          #  print("2x2x2 cell", get_coord_num())

    #  struc = Structure(lattice=atoms.cell, species=atoms.get_chemical_symbols(), coords=atoms.get_positions())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Give something ...")
    parser.add_argument("-trj_path", type=str, required=True, help="..")
    parser.add_argument("-interval", type=int, required=False, default=1, help="..")
    parser.add_argument("-nproc", type=int, required=True, help="..")
    args = parser.parse_args()
    #  nn = MinimumDistanceNN()
    #  nn = BrunnerNN_real(cutoff=12)
    #  nn = BrunnerNN_reciprocal()
    #  nn = CrystalNN(search_cutoff=12)

    nn = CrystalNN(search_cutoff=12)
    idxes = slice(0, -1, args.interval)
    atoms_list = read(args.trj_path, index=idxes, parallel=True)

    #  db = connect(db_path)
    coord_nums = []
    frames = []
    #  len_trj = len(atoms_list)
    #  with Pool(args.nproc) as pool:
    #      # implementation of  multiprocessor in tqdm. Ref.https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
    #      for result in tqdm.tqdm(pool.imap_unordered(func=task, iterable=range(len_trj)), total=len_trj):
    #          if result:
    #              frames.append(result[0])
    #              coord_nums.append(result[1])

    for i, atoms in enumerate(atoms_list):

        Al_index = [atom.index for atom in atoms if atom.symbol == "Al"]
        center_atom_i = Al_index[0]


        coord_num = get_coord_num(nn, atoms, center_atom_i, replica=1)
        if coord_num <=1:
            coord_num = get_coord_num(nn, atoms, center_atom_i, replica=2)

        frames.append(i)
        coord_nums.append(coord_num)

    import pandas as pd
    df = pd.DataFrame()
    df["Frames"] = frames
    df["CoordNum"] = coord_nums
    df.to_csv("coord_nums.csv")

    #  import matplotlib.pyplot as plt
#  plt.plot(range(len(coord_nums)), np.array(coord_nums))
#  plt.savefig("coord_num_v0.png")

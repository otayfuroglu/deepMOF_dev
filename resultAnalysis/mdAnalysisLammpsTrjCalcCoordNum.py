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

def lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair):
    symbols = [atom_type_symbol_pair[key] for key in lammps_atoms.get_atomic_numbers()]
    return Atoms(symbols=symbols, positions=lammps_atoms.positions, cell=lammps_atoms.cell)


def lammsTrj2AseDb(lammps_trj, db_path):

    if os.path.exists(db_path):
        os.remove(db_path)
    db = connect(db_path)
    atom_type_symbol_pair = {1:"Al", 2:"Li", 3:"H"}
    for lammps_atoms in lammps_trj:
        atoms = lammps2AseAtoms(atoms, atom_type_symbol_pair)   #  atoms.cell = lammps_atoms.cell
        db.write(atoms)


def task(idx):
    Path(f"tmp/task_{idx}").mkdir(parents=True, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(f"tmp/task_{idx}")

    #  atoms = db.get_atoms(idx+1)
    lammps_atoms = lammps_trj[idx]
    atoms = lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair)
    Al_index = [atom.index for atom in atoms if atom.symbol == "Al"]
    center_atom_i = Al_index[0]


    coord_num = get_coord_num(nn, atoms, center_atom_i, replica=1)
    if get_coord_num <=1:
        coord_num = get_coord_num(nn, atoms, center_atom_i, replica=2)
    os.chdir(cwd)
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
    lammps_trj_path = args.trj_path
    #  nn = MinimumDistanceNN()
    #  nn = BrunnerNN_real(cutoff=12)
    #  nn = BrunnerNN_reciprocal()
    #  nn = CrystalNN(search_cutoff=12)

    nn = CrystalNN(search_cutoff=12)
    idxes = slice(0, -1, args.interval)
    atom_type_symbol_pair = {1:"Al", 2:"Li", 3:"H"}
    #  lammps_trj_path = "../alanates/cscs/nnp_train_on16kdata_nvt_02timestep_1500K_2ns/alanates_1Bar_1500K.lammpstrj"
    lammps_trj = read(lammps_trj_path, format="lammps-dump-text", index=idxes, parallel=True)
    #  db_path = f"{lammps_trj_path.split('/')[-1].replace('.lammpstrj', '')}.db"
    #  if not os.path.exists(db_path):
    #      lammps_trj = read(lammps_trj_path, format="lammps-dump-text", index=idxes, parallel=True)
    #      lammsTrj2AseDb(lammps_trj, db_path)

    #  db = connect(db_path)
    coord_nums = []
    frames = []
    len_trj = len(lammps_trj)
    with Pool(args.nproc) as pool:
        # implementation of  multiprocessor in tqdm. Ref.https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
        for result in tqdm.tqdm(pool.imap_unordered(func=task, iterable=range(len_trj)), total=len_trj):
            if result:
                frames.append(result[0])
                coord_nums.append(result[1])

    import pandas as pd
    df = pd.DataFrame()
    df["Frames"] = frames
    df["CoordNum"] = coord_nums
    df.to_csv("coord_nums.csv")

    #  import matplotlib.pyplot as plt
    #  plt.plot(range(len(coord_nums)), np.array(coord_nums))
    #  plt.savefig("coord_num_v0.png")

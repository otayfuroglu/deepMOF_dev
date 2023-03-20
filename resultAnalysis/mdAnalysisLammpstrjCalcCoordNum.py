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

from multiprocessing import Pool
import itertools
#


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
        nn_distances.append(min(distances))
    return nn_distances


def get_coord_num(nn, struc, center_atom_i):
    coord_num = 0
    site_idexes = []
    nn_distances = get_1NN2NN_distances(nn, struc, center_atom_i)
    #  if nn_distances[1] == 0:
    #      return get_coord_numByWeigth(struc)
    for shel_i in [1, 2]:
        for site in nn.get_nn_shell_info(struc, center_atom_i, shel_i):
            site_i = site["site_index"]
            if not site_i in site_idexes:
                site_idexes.append(site_i)
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


def lammsTrj2AseDb(lammps_trj, db_path):

    if os.path.exists(db_path):
        os.remove(db_path)
    db = connect(db_path)
    atom_type_symbol_pair = {1:"Al", 2:"Li", 3:"H"}
    for lammps_atoms in lammps_trj:
        symbols = [atom_type_symbol_pair[key] for key in lammps_atoms.get_atomic_numbers()]
        atoms = Atoms(symbols=symbols, positions=lammps_atoms.positions, cell=lammps_atoms.cell)
        #  atoms.cell = lammps_atoms.cell
        db.write(atoms)


def task(idx):
    lammps_trj_path = "../alanates/cscs/nnp_train_on16kdata_nvt_02timestep_1500K_2ns/alanates_1Bar_1500K.lammpstrj"
    db_path = f"{lammps_trj_path.split('/')[-1].replace('.lammpstrj', '')}.db"
    db = connect(db_path)
    #  proc_id = os.getpid()
    Path(f"tmp/task_{idx}").mkdir(parents=True, exist_ok=True)

    #  idxes = slice(0, 49000, 100)
    #  if idx == 0:
        #  global lammps_trj
        #  global nn, atom_type_symbol_pair
    nn = CrystalNN(search_cutoff=12)
    #  atom_type_symbol_pair = {1:"Al", 2:"Li", 3:"H"}
        #  lammps_trj = read("../alanates/cscs/nnp_train_on16kdata_nvt_02timestep_1500K_2ns/alanates_1Bar_1500K.lammpstrj", format="lammps-dump-text", index=idxes, parallel=True)

    cwd = os.getcwd()
    os.chdir(f"tmp/task_{idx}")

    atoms = db.get_atoms(idx+1)
    Al_index = [atom.index for atom in atoms if atom.symbol == "Al"]
    center_atom_i = Al_index[0]


    fl_name = "test"
    write(f"{fl_name}.cif", atoms)
    site_idx = 0 #index of atom to get coordination environment
    struc = Structure.from_file(f"{fl_name}.cif") #read in CIF as Pymatgen Structure
    #  struc.make_supercell([2, 2, 2])
    coord_num = get_coord_num(nn, struc, center_atom_i)
    os.chdir(cwd)
    return coord_num
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

if __name__ == '__main__':
    #  nn = MinimumDistanceNN()
    #  nn = BrunnerNN_real(cutoff=12)
    #  nn = BrunnerNN_reciprocal()
    #  nn = CrystalNN(search_cutoff=12)
        #  atoms = read(f"./Top20/LiAlH4/isolated/{fl_name}")
    coord_nums = []
    idxes = slice(0, 49000, 100)
    #  atom_type_symbol_pair = {1:"Al", 2:"Li", 3:"H"}
    lammps_trj_path = "../alanates/cscs/nnp_train_on16kdata_nvt_02timestep_1500K_2ns/alanates_1Bar_1500K.lammpstrj"
    db_path = f"{lammps_trj_path.split('/')[-1].replace('.lammpstrj', '')}.db"
    if not os.path.exists(db_path):
        lammps_trj = read(lammps_trj_path, format="lammps-dump-text", index=idxes, parallel=True)
        lammsTrj2AseDb(lammps_trj, db_path)

    db = connect(db_path)
    len_db = db.count()
    with Pool(32) as pool:
        # implementation of  multiprocessor in tqdm. Ref.https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
        for result in tqdm.tqdm(pool.imap_unordered(func=task, iterable=range(len_db)), total=len_db):
            coord_nums.append(result)

    import matplotlib.pyplot as plt
    plt.plot(range(len(coord_nums)), np.array(coord_nums))
    plt.show()

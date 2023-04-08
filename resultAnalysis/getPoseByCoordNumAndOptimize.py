#
from ase.io.vasp import read_vasp_out
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

import os
import tqdm
import numpy as np

from numba import jit, float32
import argparse



def getCellVolume():
    _DIR = "/Users/omert/Desktop/alanates/results_top20_opt/LiAlH4"
    #  _DIR = "/Users/omert/Desktop/alanates/Top20/LiAlH4"
    #  fl = open("volumes.csv", "w")
    #  fl.write("FileNames,Volume_A^3\n")
    for root, dirs, files in os.walk(_DIR):
        if len(files) == 0:
            continue
        outcar_path = root + "/OUTCAR"
        dir_name = root.split(os.sep)[-2]
        file_name = root.split(os.sep)[-1]
        atoms = read_vasp_out(outcar_path)

        #  for fl in files:
        #      atoms = read(f"{root}/{fl}")
        #      write(f"LiAlH4_initial_structures/{dir_name}_{os.path.splitext(fl)[0]}.cif", atoms)

        #  write(f"vasp_opt_geoms/{dir_name}_{file_name}.cif", atoms)
        #  quit()
        with open(f"{dir_name}_volumes.csv", "a") as fl:
            print(f"{file_name},{len(atoms)},{atoms.get_volume()}", file=fl)
        #  break


@jit(float32(float32, float32, float32))
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


def get_coord_num(nn, struc, center_atom_i):
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


def run_get_coord_num():
    nn = CrystalNN(search_cutoff=12)
    center_atom_i = 2 # for Al atom index
    #  Al_index = [atom.index for atom in atoms if atom.symbol == "Al"]
    #  center_atom_i = Al_index[0]
    fl = open(f"coord_nums.csv", "w")
    for root, dirs, files in os.walk("./vasp_opt_geoms"):
        for file_name in files:
            print(file_name)
            struc = Structure.from_file(f"{root}/{file_name}") #read in CIF as Pymatgen Structure
            coord_num = get_coord_num(nn, struc, center_atom_i)
            if coord_num == 1:
                struc.make_supercell([2, 2, 2])
                coord_num = get_coord_num(nn, struc, center_atom_i)
            print(f"{file_name},{coord_num}", file=fl)


def lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair):
    symbols = [atom_type_symbol_pair[key] for key in lammps_atoms.get_atomic_numbers()]
    return Atoms(symbols=symbols, positions=lammps_atoms.positions, cell=lammps_atoms.cell)


def get_pose():
    interval = 1000
    idxes = slice(0, -1, interval)
    atom_type_symbol_pair = {1:"Al", 2:"Li", 3:"H"}
    lammps_trj_path = "../scicore/nnp_train_on26kdata_npt_02timestep_500K_v2_triTdamp100Pdamp100/alanates_1Bar_500K.lammpstrj"
    lammps_trj = read(lammps_trj_path, format="lammps-dump-text", index=idxes, parallel=True)

    nn = CrystalNN(search_cutoff=12)
    center_atom_i = 2 # for Al atom index
    k, l = 0, 0
    fl = open(f"coord_nums.csv", "w")
    opt_fl = open(f"opt_coord_nums.csv", "w")
    for i, lammps_atoms in enumerate(lammps_trj):
        atoms = lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair)
        #  Al_index = [atom.index for atom in atoms if atom.symbol == "Al"]


        fl_name = "test"
        write(f"{fl_name}.cif", atoms)
        struc = Structure.from_file(f"{fl_name}.cif") #read in CIF as Pymatgen Structure
        #  struc.make_supercell([2, 2, 2])
        coord_num = get_coord_num(nn, struc, center_atom_i)
        print(coord_num)

        if k <= 20 and coord_num > 5.3:
            print(f"polymeric_{i},{coord_num}", file=fl)
            #  write(f"./selectedByCoordNum/polymeric_{i}.cif", atoms)
            #NOTE add vasp opt
            print(f"opt_polymeric_{i},{coord_num}", file=opt_fl)
            k += 1
        elif l <= 20 and coord_num < 4.5:
            print(f"isolated_{i},{coord_num}", file=fl)
            #  write(f"./selectedByCoordNum/isolated_{i}.cif", atoms)
            #NOTE add vasp opt
            print(f"opt_isolated_{i},{coord_num}", file=opt_fl)
            l += 1


#  parser = argparse.ArgumentParser(description="Give something ...")
#  parser.add_argument("-trj_path", type=str, required=True, help="..")
#  parser.add_argument("-interval", type=int, required=False, default=1, help="..")
#  args = parser.parse_args()
#
#  lammps_trj_path = args.trj_path
#  interval = args.interval
get_pose()



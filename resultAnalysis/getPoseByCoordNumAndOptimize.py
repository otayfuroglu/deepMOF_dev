#
from ase.io.vasp import read_vasp_out
from ase import Atoms
from ase.io import read, write
from ase.db import connect
from ase.neighborlist import (NeighborList,
                              NewPrimitiveNeighborList,
                              PrimitiveNeighborList,
                              natural_cutoffs)
from ase.calculators.vasp import Vasp
from pymatgen.core import Structure
from pymatgen.analysis.local_env import  (BrunnerNN_reciprocal,
                                          BrunnerNN_real,
                                          CrystalNN,
                                          MinimumDistanceNN)

from pathlib import Path
import os
import tqdm
import numpy as np
import pandas as pd

from numba import jit, float32
import argparse



def setVaspCalculator(calc):
    calc.set(
     system = f"self_energy_R2SCAN",
     prec = 'Accurate',
     ismear = 3, # gaussian smearing
     sigma = 0.02,
     isym = -1,
     #  isym = 2,
     symprec = 1.0e-06, # to overcome Inconsistent Bravais lattice types found for crystalline error
     nelm = 60, # max number of scf iterations
     lreal = "AUTO", # recommended by vasp for more than 30 atoms
     isif = 3, # calculate stress and optimize cell shape and volume
     ibrion = 2, #2 # use 2 for conj. grad.
     enaug = 800, # test using the default here!                    <--------
     addgrid = True, # reduces force noise, but vasp says do not this as default in all computations? <--------
     algo = "Normal", # recommended for GPU (maybe also try Fast ?)                                   <--------
     metagga = "R2SCAN",
     lasph = True, # recommended for metaggas in the vasp wiki, increases accuracy
     kpar = 1, # set to numbers of gpus
     ncore = 4, # for gpu
     nsim = 1, # ###### 1 was used for many calculations
     ediff = 1.0e-07, # changed from 1.0e-07
     ediffg = -0.005,
     encut = 800,
     #  encut = 500, # for system which contain Ca or K elements
     nsw = nsw, #0 #600 # number of optimization steps
     kspacing = 0.2,
     kgamma   = True, # from emergence paper (should converge faster)
     pstress = 0, # pressure in MD
     lwave = False, # dont save wavefunction for single point run
     lcharg = False, # same for charge
    )



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



def lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair):
    symbols = [atom_type_symbol_pair[key] for key in lammps_atoms.get_atomic_numbers()]
    return Atoms(symbols=symbols, positions=lammps_atoms.positions, cell=lammps_atoms.cell)


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-trj_path", type=str, required=True, help="..")
parser.add_argument("-csv_path", type=str, required=True, help="..")
parser.add_argument("-coord_type", type=str, required=True, help="..")
#  parser.add_argument("-skip", type=int, required=False, default=0, help="..")
parser.add_argument("-IDX", type=int, required=True, default=1, help="..")
args = parser.parse_args()

df = pd.read_csv(args.csv_path)

coord_type = args.coord_type
if coord_type == "polymeric":
    df = df.loc[df["CoordNum"] > 5.9].reset_index()
    f_idx = int(len(df)/25) * args.IDX  # to select homojenuosly
    idx = int(df["Frames"][f_idx].split("_")[-1])
    coord_num = df["CoordNum"][f_idx]
elif coord_type == "isolated":
    df = df.loc[df["CoordNum"] < 4.0].reset_index()
    f_idx = int(len(df)/25) * args.IDX  # to select homojenuosly
    idx = int(df["Frames"][f_idx].split("_")[-1])
    coord_num = df["CoordNum"][f_idx]
else:
    print("Enter coord_type correctly")
    quit(1)

atom_type_symbol_pair = {1:"Al", 2:"Li", 3:"H"}
#  lammps_trj_path = "../scicore/nnp_train_on26kdata_npt_02timestep_500K_v2_triTdamp100Pdamp100/alanates_1Bar_500K.lammpstrj"
lammps_trj = read(args.trj_path, format="lammps-dump-text", index=":", parallel=True)

fl = open(f"{coord_type}_coord_nums.csv", "a")
print(f"{coord_type}_{idx},{coord_num}", file=fl, flush=True)
fl.close()

opt_fl = open(f"opt_{coord_type}_coord_nums.csv", "a")
selected_dir = "selectedByCoordNum"
Path(f"{selected_dir}").mkdir(parents=True, exist_ok=True)
Path(f"vasp_calc/{idx}").mkdir(parents=True, exist_ok=True)

cwd = os.getcwd()

# set VASP calc objects
calc = Vasp()
nsw = 100
setVaspCalculator(calc)

lammps_atoms = lammps_trj[idx]
atoms = lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair)
write(f"./selectedByCoordNum/{coord_type}_{idx}.cif", atoms)

os.chdir(f"vasp_calc/{idx}")
atoms.pbc = [1, 1, 1]
atoms.calc = calc
atoms.get_potential_energy()

from mdAnalysisLammpsTrjCalcCoordNum import get_coord_num

nn = CrystalNN(search_cutoff=12)

Al_index = [atom.index for atom in atoms if atom.symbol == "Al"]
center_atom_i = Al_index[0]

opt_coord_num = get_coord_num(nn, atoms, center_atom_i, replica=1)
if opt_coord_num <= 1:
    opt_coord_num = get_coord_num(nn, atoms, center_atom_i, replica=2)

os.chdir(cwd)

write(f"./selectedByCoordNum/opt_{coord_type}_{idx}.cif", atoms)
print(f"opt_{coord_type}_{idx},{opt_coord_num}", file=opt_fl, flush=True)
opt_fl.close()


#  for i, lammps_atoms in enumerate(lammps_trj):
#      atoms = lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair)
#      #  Al_index = [atom.index for atom in atoms if atom.symbol == "Al"]
#
#
#      #  fl_name = "test"
#      #  write(f"{fl_name}.cif", atoms)
#      #  struc = Structure.from_file(f"{fl_name}.cif") #read in CIF as Pymatgen Structure
#      #  struc.make_supercell([2, 2, 2])
#      coord_num = get_coord_num(nn, atoms, center_atom_i)
#      #  print(coord_num)
#
#      if k <= 20 and coord_num > 5.4:
#          print(f"polymeric_{i},{coord_num}", file=fl)
#          write(f"./selectedByCoordNum/polymeric_{i}.cif", atoms)
#
#          os.chdir("vasp_calc")
#          atoms.pbc = [1, 1, 1]
#          atoms.calc = calc
#          atoms.get_potential_energy()
#          os.chdir(cwd)
#
#          coord_num = get_coord_num(nn, atoms, center_atom_i)
#          print(f"opt_polymeric_{i},{coord_num}", file=opt_fl)
#          k += 1
#      elif l <= 20 and coord_num < 4.5:
#          print(f"isolated_{i},{coord_num}", file=fl)
#          write(f"./selectedByCoordNum/isolated_{i}.cif", atoms)
#
#          os.chdir("vasp_calc")
#          atoms.pbc = [1, 1, 1]
#          atoms.calc = calc
#          atoms.get_potential_energy()
#          os.chdir(cwd)
#
#          coord_num = get_coord_num(nn, atoms, center_atom_i)
#          print(f"opt_isolated_{i},{coord_num}", file=opt_fl)
#          l += 1
#
#      if k == 20 and l == 20:
#          break



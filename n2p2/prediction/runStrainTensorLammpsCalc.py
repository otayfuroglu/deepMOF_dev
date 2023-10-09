#
#
from ase.constraints import StrainFilter, UnitCellFilter
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS

import os
import argparse

import numpy as np

#  from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.lammpsrun import LAMMPS


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-geoms_dir", type=str, required=True, help="..")
parser.add_argument("-model_dir", type=str, required=True, help="..")
parser.add_argument("-best_epoch", type=int, required=True, help="..")
parser.add_argument("-energy_u", type=str, required=True, help="..")
parser.add_argument("-length_u", type=str, required=True, help="..")

args = parser.parse_args()

parameters = {
    'units': 'metal',
    "atom_style": "atomic",
    'pair_style': ' nnp dir n2p2_parameters  maxew 30000 cflength 1.8897261328 cfenergy 0.0367493254  emap "1:Al,2:Li,3:H"',
    'pair_coeff': ['* * 6.5']}

#  calc = LAMMPSlib(lmpcmds=cmds, log_file='lammps.log')
calc = LAMMPS(parameters=parameters, specorder=["Al", "Li", "H"], tmp_dir="tmp")

file_names = [fl for fl in os.listdir(args.geoms_dir) if ".extxyz" in fl]
print(len(file_names))

for i, file_name in enumerate(file_names):
    #  if "beta" not in file_name:
        #  continue

    atoms = read(f"{args.geoms_dir}/{file_name}")
    atoms.pbc = True
    atoms.calc = calc

    traj = Trajectory(f"{args.geoms_dir.split('/')[-1]}_{file_name.split('.')[0]}_strainTenso.traj" , "w", atoms)

    sf = StrainFilter(atoms)

    opt = BFGS(sf)
    opt.attach(traj)
    opt.run(0.0001)


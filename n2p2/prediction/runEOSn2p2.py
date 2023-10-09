from ase.calculators.vasp import Vasp
from ase.io.extxyz import read_extxyz
from ase.io import read
from ase import Atoms
from ase.io.trajectory import Trajectory

from pathlib import Path
import os
import argparse

import numpy as np

from n2p2AseInterFace import n2p2Calculator


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-geoms_dir", type=str, required=True, help="..")
parser.add_argument("-model_dir", type=str, required=True, help="..")
parser.add_argument("-best_epoch", type=int, required=True, help="..")
parser.add_argument("-energy_u", type=str, required=True, help="..")
parser.add_argument("-length_u", type=str, required=True, help="..")

args = parser.parse_args()


calc = n2p2Calculator(model_dir=args.model_dir,
                      best_epoch=args.best_epoch,
                      energy_units=args.energy_u,
                      length_units=args.length_u,
                     )

file_names = [fl for fl in os.listdir(args.geoms_dir) if ".extxyz" in fl]
print(len(file_names))

for i, file_name in enumerate(file_names):
    #  if "beta" not in file_name:
        #  continue

    atoms = read(f"{args.geoms_dir}/{file_name}")
    atoms.pbc = True
    atoms.calc = calc

    traj = Trajectory(f"{args.geoms_dir.split('/')[-1]}_{file_name.split('.')[0]}_EOS.traj" , "w")
    scaleFs = np.linspace(0.95, 1.10, 8)
    cell = atoms.get_cell()

    print("Starting EOS calculations")
    print("Number of scaling factor values:", len(scaleFs))
    for scaleF in scaleFs:
        atoms.set_cell(cell * scaleF, scale_atoms=True)
        print(atoms.get_potential_energy())
        traj.write(atoms)
    #  if i == 2:
        #  break


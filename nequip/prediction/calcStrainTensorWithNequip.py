#
from ase.io.trajectory import Trajectory
from ase.io import read
from nequip.ase import NequIPCalculator

from ase.io.trajectory import Trajectory
from ase.constraints import StrainFilter
from ase.optimize import BFGS

import argparse
import os
import numpy as np


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-geoms_dir", type=str, required=True, help="..")
parser.add_argument("-model_path", type=str, required=True, help="..")
args = parser.parse_args()

calculator = NequIPCalculator.from_deployed_model(
    model_path=args.model_path,
    device="cuda",
    #  energy_units_to_eV=args.energy_units_to_eV,
    #  length_units_to_A=args.length_units_to_A,
)

for file_name in [fl for fl in os.listdir(args.geoms_dir) if ".extxyz" in fl]:

    atoms = read(f"{args.geoms_dir}/{file_name}")
    atoms.pbc = True
    atoms.calc = calculator
    traj = Trajectory(f"{args.geoms_dir.split('/')[-1]}_{file_name.split('.')[0]}_strainTensor.traj" , "w", atoms)

    sf = StrainFilter(atoms)
    opt = BFGS(sf)

    opt.attach(traj)
    opt.run(0.0001)


#
from ase.io import read, write
import os
#  from dftd4 import D4_model
from ase.calculators.orca import ORCA
from ase.constraints import FixAtoms
from ase.optimize import BFGS

#from gpaw import GPAW, PW

#  import numpy as np
import pandas as pd
import multiprocessing
from orca_parser import OrcaParser
import argparse

from pathlib import Path
from calculateGeomWithQM import (prepareDDECinput,
                                 orca_calculator)

from orca_io import (read_orca_h_charges,
                     read_orca_chelpg_charges,
                     read_orca_ddec_charges,
                    )




parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-geoms_path", type=str, required=True)
parser.add_argument("-orca_path", type=str, required=True)
parser.add_argument("-calc_type", type=str, required=True)
#  parser.add_argument("-idx", type=int, required=True)
parser.add_argument("-n_task", type=int, required=True)
args = parser.parse_args()

geoms_path = args.geoms_path
base = geoms_path.split("/")[-1].split(".")[0]

calc_type = args.calc_type
n_task = args.n_task
orca_path = args.orca_path
#  idx = args.idx
idx = 0

atoms = read(geoms_path, index=-1)
try:
    label = atoms.info["label"]
except:
    label = "frame_" + str(idx)


OUT_DIR = Path(f"{calc_type}_{base}/{label}")

if not os.path.exists(OUT_DIR):
    #  OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True)

    # change to local scratch directory
    #  os.chdir(TMP_DIR)

    cwd = os.getcwd()
    os.chdir(OUT_DIR)

    atoms.calc = orca_calculator(orca_path, label, calc_type="engrad", n_task=n_task)

    indices = [atom.index for atom in atoms]

    c = FixAtoms(indices=indices[:-3])
    atoms.set_constraint(c)

    dyn = BFGS(atoms)
    dyn.run(fmax=0.005)
    #  print(charges)
    os.chdir(cwd)
    write(f"{calc_type}_{base}.extxyz", atoms, append=True)




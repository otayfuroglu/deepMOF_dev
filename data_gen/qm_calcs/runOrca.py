
#
from ase.io import read, write
import os
#  from dftd4 import D4_model
from ase.calculators.orca import ORCA
#from gpaw import GPAW, PW

#  import numpy as np
import pandas as pd
import multiprocessing
from orca_parser import OrcaParser
import argparse

from pathlib import Path



def orca_calculator(label, n_task, initial_gbw=['', '']):
    return ORCA(label=label,
                maxiter=400,
                charge=0, mult=1,
                orcasimpleinput='SP PBE D4 DEF2-TZVP DEF2/J RIJDX MINIPRINT NOPRINTMOS NoKeepInts NOKEEPDENS ' + initial_gbw[0],
                orcablocks='%scf Convergence tight \n maxiter 400 end \n %output \n Print[ P_Hirshfeld] 1 end \n %pal nprocs ' + str(n_task) + ' end' + initial_gbw[1]
                )


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-geoms_path", type=str, required=True)
parser.add_argument("-calc_type", type=str, required=True)
parser.add_argument("-n_task", type=int, required=True)
args = parser.parse_args()

geoms_path = args.geoms_path
label = geoms_path.split("/")[-1].split(".")[0]

calc_type = args.calc_type
n_task = args.n_task

OUT_DIR = f"{calc_type}_{label}"


Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
# change to local scratch directory
#  os.chdir(TMP_DIR)

atoms = read(geoms_path, index=0)

cwd = os.getcwd()
os.chdir(OUT_DIR)

atoms.calc = orca_calculator(label, n_task)
atoms.get_potential_energy()
#  os.chdir(cwd)
write(f"{calc_type}_{label}.extxyz", atoms)

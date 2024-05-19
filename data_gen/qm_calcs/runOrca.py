
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
from calculateGeomWithQM import (prepareDDECinput,
                                 orca_calculator)

from orca_io import (read_orca_h_charges,
                     read_orca_chelpg_charges,
                     read_orca_ddec_charges,
                    )




parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-geoms_path", type=str, required=True)
parser.add_argument("-calc_type", type=str, required=True)
parser.add_argument("-idx", type=int, required=True)
parser.add_argument("-n_task", type=int, required=True)
args = parser.parse_args()

geoms_path = args.geoms_path
base = geoms_path.split("/")[-1].split(".")[0]

calc_type = args.calc_type
n_task = args.n_task
idx = args.idx

atoms = read(geoms_path, index=idx)
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

    #  cwd = os.getcwd()
    os.chdir(OUT_DIR)

    atoms.calc = orca_calculator(label, calc_type, n_task)
    atoms.get_potential_energy()

    h_charges = read_orca_h_charges(f"{label}.out")
    atoms.arrays["HFPQ"] = h_charges

    chelpg_charges = read_orca_chelpg_charges(f"{label}.pc_chelpg")
    atoms.arrays["CHELPGPQ"] = chelpg_charges

    prepareDDECinput(label)
    os.system("/arf/home/otayfuroglu/miniconda3/pkgs/chargemol-3.5-h1990efc_0/bin/chargemol")
    ddec_charges = read_orca_ddec_charges("DDEC6_even_tempered_net_atomic_charges.xyz")
    atoms.arrays["DDECPQ"] = ddec_charges
    #  print(charges)
    #  os.chdir(cwd)
    write(f"{calc_type}_{base}.extxyz", atoms, append=True)




from ase.io import read, write

import os
import sys
from pathlib import Path
import argparse


import numpy as np
from ase.thermochemistry import HarmonicThermo
from ase.vibrations import Vibrations
from ase import units

from ase.optimize import BFGS

from nequip.ase import NequIPCalculator
from ase.build import make_supercell




def getOpt(atoms, calc, name):
    atoms.calc = calc
    opt = BFGS(atoms)
    opt.run(fmax=0.0001)
    atoms.info["label"] = name
    return atoms


def getVib(atoms, calc):
    atoms.calc = calc

    #  h = 1e-4 # shift to get the hessian matrix
    h = 0.01

    # evaluate hessian matrix
    vib = Vibrations(atoms, delta = h)
    vib.run()
    #  vib_energies = vib.get_energies()

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-extxyz_path", type=str, required=True, help="..")
parser.add_argument("-idx", type=int, required=True, help="..")
args = parser.parse_args()
extxyz_path = args.extxyz_path

model_path = "../model_31k.pth"
calc = NequIPCalculator.from_deployed_model(model_path=model_path, device="cuda")

atoms_list = read(extxyz_path, index=":")


if "isolated" in extxyz_path:
    keyword = "isolated"
elif "polymeric" in extxyz_path:
    keyword = "polymeric"
else:
    keyword = "test"

#  keyword += "_24atoms_4x3x4"

CWD = os.getcwd()
#  for i, atoms in enumerate(atoms_list):
atoms = atoms_list[args.idx]

try:
    name = atoms.info["label"]
except:
    name = f"structure_{i}"


WORKS_DIR = Path(f"{keyword}/{name}")
WORKS_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKS_DIR)
# to call for calculation of vibiration

#  P = [[0, 0, -4], [0, -3, 0], [-4, 0, 0]]
#  atoms = make_supercell(atoms, P)

atoms = getOpt(atoms, calc, name)

getVib(atoms, calc)

os.chdir(CWD)

write(f"nequip_opt_lowest_10_{keyword}.extxyz", atoms, append=True)



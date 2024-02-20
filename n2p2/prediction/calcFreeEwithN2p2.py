from ase.io import read, write

import os
import sys
from pathlib import Path
import argparse


import numpy as np
from ase.thermochemistry import HarmonicThermo
from ase.vibrations import Vibrations
from ase import units

from n2p2AseInterFace import n2p2Calculator
from ase.calculators.lammpsrun import LAMMPS
from ase.build import make_supercell

def getLammpsCalc(supercoder:list) -> LAMMPS:

    #  prepareModel()
    prepareModelFromWeights()

    parameters = {
        'units': 'metal',
        "atom_style": "atomic",
        'pair_style': ' nnp dir ./  maxew 30000 cflength 1.8897261328 cfenergy 0.0367493254  emap "1:Al,2:Li,3:H"',
        'pair_coeff': ['* * 6.5']}

    return LAMMPS(parameters=parameters, specorder=supercoder, tmp_dir="tmp_lammps")


def prepareModelFromWeights():
    os.system(f"cp ../../input.nn ./")
    os.system(f"cp ../../scaling.data ./")
    os.system(f"cp ../../weights* ./")


def prepareModel():
    os.system(f"cp {args.MODEL_DIR}/input.nn ./")
    os.system(f"cp {args.MODEL_DIR}/scaling.data ./")
    weights_files = [item for item in os.listdir(args.MODEL_DIR) if "weights" in item]
    best_weights_files = [item for item in weights_files if int(item.split(".")[-2]) == args.best_epoch]
    assert len(best_weights_files) != 0, "Erro: NOT FOUND best epoch number"
    for best_weights_file in best_weights_files:
        os.system(f"cp {args.MODEL_DIR}/{best_weights_file} ./{best_weights_file[:11]}.data")


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
args = parser.parse_args()
extxyz_path = args.extxyz_path


atoms_list = read(extxyz_path, index=":")


if "isolated" in extxyz_path:
    keyword = "isolated"
elif "polymeric" in extxyz_path:
    keyword = "polymeric"

P = [[0, 0, -1], [0, -2, 0], [-2, 0, 0]]

CWD = os.getcwd()
for i, atoms in enumerate(atoms_list):

#      if len(atoms) == 24:
#          P = [[0, 0, -1], [0, -2, 0], [-2, 0, 0]]
#
    atoms = make_supercell(atoms, P)

    WORKS_DIR = Path(f"{keyword}/structure_{i}")
    WORKS_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WORKS_DIR)
    # to call for calculation of vibiration
    calc = getLammpsCalc(supercoder=["Al", "Li", "H"])
    getVib(atoms, calc)

    os.chdir(CWD)

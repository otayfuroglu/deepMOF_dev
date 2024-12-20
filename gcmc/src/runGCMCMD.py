import sys
import os
import numpy as np
import torch
torch.set_num_threads(6)
from ase import Atoms
from ase.io import read, write
from ase.data import vdw_radii
from ase.build import make_supercell
from ase.optimize import BFGS, LBFGS
from ase.filters import UnitCellFilter
from ase.io.trajectory import Trajectory

from time import time
from gcmc_md import AI_GCMCMD
from molmod.units import *
from time import time

from nequip.ase.nequip_calculator import NequIPCalculator
from utilities import PREOS

from multiprocessing import Pool
import argparse


def getBoolStr(string):
    string = string.lower()
    if "true" in string or "yes" in string:
        return True
    elif "false" in string or "no" in string:
        return False
    else:
        print("%s is bad input!!! Must be Yes/No or True/False" %string)
        sys.exit(1)


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-pressure", type=float, required=True, help="")
parser.add_argument("-temperature", type=float, required=True, help="")
parser.add_argument("-stepsize", type=float, required=True, help="")
parser.add_argument("-totalsteps", type=int, required=True, help="")
parser.add_argument("-nmdsteps", type=int, required=True, help="")
parser.add_argument("-ngcmcsteps", type=int, required=True, help="")
parser.add_argument("-nmcmoves", type=int, required=True, help="")
parser.add_argument("-flex_ads", type=str, required=True, help="")
parser.add_argument("-opt", type=str, required=True, help="")
parser.add_argument("-model_gcmc_path", type=str, required=True, help="")
parser.add_argument("-model_md_path", type=str, required=True, help="")
parser.add_argument("-struc_path", type=str, required=True, help="")
parser.add_argument("-molecule_path", type=str, required=True, help="")
parser.add_argument("-interval", type=int, required=True, help="")
args = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load_model(model_path):
    #  Modify species for Mg-MOF-74 (see training yaml file)
    return NequIPCalculator.from_deployed_model(
        model_path = model_path, #'./MgF2_nonbonded_v10_nnp1_e10.pth',
        species_to_type_name = {"C" : "C", "H" : "H", "O" : "O", "Mg" : "Mg"}, device=device)


#  temperature = 273 * kelvin
temperature = args.temperature * kelvin
pressure = args.pressure * bar
stepsize = args.stepsize
totalsteps = args.totalsteps
nmdsteps = args.nmdsteps # invoke this fix every nmdsteps steps
ngcmcsteps = args.ngcmcsteps # average number of GCMC exchanges to attempt every nmdsteps steps
nmcmoves = args.nmcmoves # average number of MC moves to attempt every nmdsteps steps
model_gcmc_path = args.model_gcmc_path
model_md_path = args.model_md_path
struc_path = args.struc_path
molecule_path = args.molecule_path
interval = args.interval

flex_ads = getBoolStr(args.flex_ads)
opt = getBoolStr(args.opt)
# Preferably run on GPUs
device = 'cuda'

model_gcmc = load_model(model_gcmc_path)
model_md = load_model(model_md_path)

#  atoms_frame = read('MgMOF74_clean_fromCORE.cif')
atoms_frame = read(struc_path)
replica = [1, 1, 1]
P = [[0, 0, -replica[0]], [0, -replica[1], 0], [-replica[2], 0, 0]]
atoms_frame = make_supercell(atoms_frame, P)

if opt:
    atoms_frame.calc = model_md
    #  write("framebeforeopt.cif", atoms_frame)
    # geom opt frame based on model
    # to account relaxation of cell
    traj = Trajectory('frames.traj', 'w', atoms_frame)
    ucf = UnitCellFilter(atoms_frame)
    optimizer = LBFGS(ucf)
    optimizer.attach(traj)
    optimizer.run(fmax=0.001)

write("frame0.extxyz", atoms_frame)
#  write("frame0.cif", atoms_frame)

#  atoms_frame = read("frame0.cif")
#  quit()

# C and O were renamed to Co and Os to differentiate them from framework atoms during training
#  atoms_ads = read('./co2_v2.xyz')
atoms_ads = read(molecule_path)
vdw_radii = vdw_radii.copy()
#  vdw_radii[76] = vdw_radii[8]
#  vdw_radii[27] = vdw_radii[6]
# Mg radius is set to 1.0 A
vdw_radii[12] = 1.0


eos = PREOS.from_name('carbondioxide')
fugacity = eos.calculate_fugacity(temperature, pressure)

results_dir = f"gcmcmd_results_stepsize{stepsize}_N{nmdsteps}_X{ngcmcsteps+nmcmoves}_flexAds{flex_ads}_opt{opt}_{pressure/bar}bar_{int(temperature)}K"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)


gcmc_md = AI_GCMCMD(model_gcmc, model_md, results_dir, interval, atoms_frame, atoms_ads, flex_ads,
                  temperature, pressure, fugacity, device, vdw_radii)

gcmc_md.run(stepsize, totalsteps, nmdsteps, ngcmcsteps, nmcmoves)
#

import sys
import os
import numpy as np
import torch
torch.set_num_threads(6)
from ase import Atoms
from ase.io import read, write
from ase.data import vdw_radii
from ase.build import make_supercell

from time import time
from gcmc import AI_GCMC
from molmod.units import *
from time import time

from nequip.ase.nequip_calculator import NequIPCalculator
from utilities import PREOS

from multiprocessing import Pool

import argparse

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-pressure", type=float, required=True, help="")
parser.add_argument("-temperature", type=float, required=True, help="")
parser.add_argument("-model_path", type=str, required=True, help="")
parser.add_argument("-struc_path", type=str, required=True, help="")
parser.add_argument("-molecule_path", type=str, required=True, help="")
args = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Preferably run on GPUs
device = 'cuda'
#  Modify species for Mg-MOF-74 (see training yaml file)
model = NequIPCalculator.from_deployed_model(model_path = args.model_path, #'./MgF2_nonbonded_v10_nnp1_e10.pth',
                                                    species_to_type_name = {"C" : "C",
                                                                            "H" : "H",
                                                                            "O" : "O",
                                                                            "Mg" : "Mg",
                                                                            #  "Os" : "Os",
                                                                            #  "Co" : "Co"
                                                                           },
                                                    device=device)

#  atoms_frame = read('MgMOF74_clean_fromCORE.cif')
atoms_frame = read(args.struc_path)

replica = [1, 1, 1]
P = [[0, 0, -replica[0]], [0, -replica[1], 0], [-replica[2], 0, 0]]
atoms_frame = make_supercell(atoms_frame, P)
write("frame0.extxyz", atoms_frame)

atoms_frame.calc = model
# C and O were renamed to Co and Os to differentiate them from framework atoms during training
#  atoms_ads = read('./co2_v2.xyz')
atoms_ads = read(args.molecule_path)
vdw_radii = vdw_radii.copy()
#  vdw_radii[76] = vdw_radii[8]
#  vdw_radii[27] = vdw_radii[6]
# Mg radius is set to 1.0 A
vdw_radii[12] = 1.0

#  temperature = 273 * kelvin
temperature = args.temperature * kelvin
pressure = args.pressure * bar
#  print(pressure)
#  pressure = 0.1 * bar

eos = PREOS.from_name('carbondioxide')
fugacity = eos.calculate_fugacity(temperature, pressure)

results_dir = f"test_results_{pressure/bar}bar_{temperature}K"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

interval = 50

gcmc = AI_GCMC(model, results_dir, interval, atoms_frame, atoms_ads, temperature, pressure, fugacity, device, vdw_radii)
#  gcmc.run(6000000)
gcmc.run(2000000)


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

from utilities import PREOS


#  from dynamicChargeFF import ForceField
from dynamicChargeFFforGCMC import ForceField
from ChargePredict import ChargeCalculator


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Preferably run on GPUs
device = 'cuda'

atoms_frame = read('MgMOF74_clean_fromCORE.cif')

replica = [1, 1, 1]
P = [[0, 0, -replica[0]], [0, -replica[1], 0], [-replica[2], 0, 0]]
atoms_frame = make_supercell(atoms_frame, P)

nn1 = 200
nn2 = 160
chargeCalc = ChargeCalculator(hidden_size1=nn1,hidden_size2=nn2)

#  struct = read("./nan_test.extxyz")
#  charges = chargeCalc.get_charge(struct)
#  for ch in charges:
#      if np.isnan(ch):
#          print(ch)
#  quit()

calc = ForceField(chargeCalc, rc=12)
atoms_frame.calc = calc

#  print(atoms_frame.get_potential_energy())
#  quit()

#  atoms_ads = read('./co2_v2.xyz')
atoms_ads = read('co2.xyz')
atoms_ads.calc = calc
vdw_radii = vdw_radii.copy()
# Mg radius is set to 1.0 A
vdw_radii[12] = 1.0

#  T = 273 * kelvin
T = 298 * kelvin
P = 1.0 * bar
#  P = 0.1 * bar

eos = PREOS.from_name('carbondioxide')
fugacity = eos.calculate_fugacity(T,P)

results_dir = f"test4_elecPot_results_{P/bar}bar_{T}K"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

interval = 10
gcmc = AI_GCMC(calc, results_dir, interval, atoms_frame, atoms_ads, T, P, fugacity, device, vdw_radii)
#  gcmc.run(6000000)
gcmc.run(10000)


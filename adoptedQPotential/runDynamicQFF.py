#

import torch
from ase.io import read, write
import tqdm
import argparse


import numpy as np
import torch
torch.set_num_threads(6)

from dynamicQFF import ForceField
from ChargePredict import ChargeCalculator


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-extxyz_path", type=str, required=True, help="")
args = parser.parse_args()

device = "cuda"

nn1 = 200
nn2 = 160
chargeCalc = ChargeCalculator(hidden_size1=nn1,hidden_size2=nn2)
calc = ForceField(chargeCalc, rc=12)

extxyz_path = args.extxyz_path
file_base = extxyz_path.split("/")[-1].split(".")[0]


atoms_list = read(extxyz_path, index=":")

fl_enegies = open(f"{file_base}_model_energeis.csv", "w")
fl_enegies.write(f"index,e_Model\n")

#  while n_sample <= 250:
for i in tqdm.trange(0, len(atoms_list), 1):
    #  for i in range(0, len(atoms_list), 1):

    atoms = atoms_list[i]
    label = atoms.info["label"]

    atoms.calc = calc
    model_energy = atoms.get_potential_energy()# / len(atoms)
    write(f"engrad_dynamicQFF{file_base}.extxyz", atoms, append=True)


    #  diff_forces = abs(qm_forces - model_forces)

    fl_enegies.write(f"{label},{model_energy}\n")
    fl_enegies.flush()
    fl_enegies.flush()



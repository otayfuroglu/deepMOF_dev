#!/bin/env python

from __future__ import absolute_import
from __future__ import print_function
import pynnp
import numpy as np
#  from ase.db import connect
from ase.io import read
from ase import Atoms

import os, shutil
import torch
import tqdm

from n2p2AseInterFace import n2p2Calculator
import argparse

from multiprocessing import Pool, current_process


def get_fmax_idx(forces):
    """
    Args:
    forces(3D torch tensor)

    retrun:
    maximum force conponent indices (zero 1D torch tensor)
    """

    if not torch.is_tensor(forces):
        forces = torch.from_numpy(forces) # if numpy array convert to 3d tensor
    else:
        forces = forces.squeeze(0)

    abs_forces = forces.abs()
    abs_idxs = (abs_forces==torch.max(abs_forces)).nonzero().squeeze(0) # get index of max value in abs_forces.
    if len(abs_idxs.shape) > 1: # if there are more than one max value
        abs_idxs = abs_idxs[0]  # select just one
    return abs_idxs


def get_fmax_componentFrom_idx(forces, fmax_component_idx):
    """
    xxx
    """

    if not torch.is_tensor(forces):
        forces = torch.from_numpy(forces) # if numpy array convert to 3d tensor
    else:
        forces = forces.squeeze(0)

    return forces[fmax_component_idx[0], fmax_component_idx[1]].item()


def runPredict(idx):

    #  try:
        current = current_process()
        proc_dir = RESULT_DIR + "/tmp_%s" % current._identity
        if not os.path.exists(proc_dir):
            os.mkdir(proc_dir)
        os.chdir(proc_dir)

        atoms = atoms_list[idx]
        qm_energy = atoms.get_potential_energy()
        qm_forces = atoms.get_forces()
        # first get fmax indices than get qm_fmax and n2p2 fmax component
        qm_fmax_component_idx = get_fmax_idx(qm_forces)
        qm_fmax_component = get_fmax_componentFrom_idx(qm_forces,
                                                       qm_fmax_component_idx)

        calculator = n2p2Calculator(model_dir=MODEL_DIR,
                                    best_epoch=args.best_epoch,
                                    energy_units=args.energy_u,
                                    length_units=args.length_u,
                                   )
        if atoms.cell:
            atoms.pbc = True
        atoms.set_calculator(calculator)

        #  n2p2_energy = atoms.get_potential_energy()
        n2p2_energy = atoms.get_potential_energy()
        n2p2_forces = atoms.get_forces()
        #  print(n2p2_energy, qm_energy)
        n2p2_fmax_component = get_fmax_componentFrom_idx(n2p2_forces, qm_fmax_component_idx)

        return (idx, len(atoms), qm_energy,
                n2p2_energy, qm_fmax_component,
                n2p2_fmax_component, qm_forces, n2p2_forces)
    #  except:
    #      return None


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-data_path", "--data_path",
                    type=str, required=True,
                    help="..")
parser.add_argument("-energy_u", "--energy_u",
                    type=str, required=True,
                    help="..")
parser.add_argument("-length_u", "--length_u",
                    type=str, required=True,
                    help="..")
parser.add_argument("-MODEL_DIR", "--MODEL_DIR",
                    type=str, required=True,
                    help="..")
parser.add_argument("-RESULT_DIR", "--RESULT_DIR",
                    type=str, required=True,
                    help="..")
parser.add_argument("-best_epoch", "--best_epoch",
                    type=int, required=True,
                    help="..")
parser.add_argument("-nproc", "--nproc",
                    type=int, required=True,
                    help="..")
args = parser.parse_args()

MODEL_DIR = args.MODEL_DIR
RESULT_DIR = args.RESULT_DIR
data_path = args.data_path


#  os.chdir(RESULT_DIR)
#  db = connect(data_path)
#  len_db = db.count()
#  db = db.select()

atoms_list = read(data_path, index=":")
len_db = len(atoms_list)

keyword = data_path.split("/")[-1].split(".")[0]

fl_energy = open(f"{RESULT_DIR}/{keyword}_energy.csv", "w")
fl_energy.write("Name,qmEnergyPa(eV),modelEnergyPa(eV)\n")

fl_max_force = open(f"{RESULT_DIR}/{keyword}_max_force.csv", "w")
fl_max_force.write("Name,qmMaxForce(eV),modelMaxForce(eV)\n")

fl_forces = open(f"{RESULT_DIR}/{keyword}_forces.csv", "w")
fl_forces.write("Name,qmForces(eV/A),modelForces(eV/A)\n")

fl_summary = open(f"{RESULT_DIR}/{keyword}_diff.csv", "w")
fl_summary.write("Name,DifEnergy(eV/A),DifEnergyPa(eV),diffFmaxComp(eV/A)\n")

#  for idx in range(len_db):
    #  runPredict_test(idx)
    #  quit()

pool = Pool(processes=args.nproc)
result_list_tqdm = []
# implementation of  multiprocessor in tqdm. Ref.https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
for result in tqdm.tqdm(pool.imap_unordered(func=runPredict, iterable=range(len_db)), total=len_db):
    if result:
        #  result_list_tqdm.append(result)
        (idx, n_atoms, qm_energy,
         n2p2_energy, qm_fmax_component,
         n2p2_fmax_component, qm_forces, n2p2_forces) = result

        diffE = qm_energy - n2p2_energy
        diffEpa = diffE / n_atoms

        diffFamxComp = n2p2_fmax_component - qm_fmax_component
        print(idx, ",", qm_energy/n_atoms, ",", n2p2_energy/n_atoms, file=fl_energy)
        print(idx, ",", qm_fmax_component, ",", n2p2_fmax_component, file=fl_max_force)
        print(idx, ",", diffE, ",", diffEpa, ",", diffFamxComp, file=fl_summary)
        for i in range(len(qm_forces.flatten())):
            print(idx, ",", qm_forces.flatten()[i], ",", n2p2_forces.flatten()[i], file=fl_forces)


        fl_energy.flush()
        fl_max_force.flush()
        fl_forces.flush()
        #  fl_summary.flush()

fl_energy.close()
fl_max_force.close()
fl_forces.close()
fl_summary.close()

# remove temporary directories
for tmp_dir in [tmp_dir for tmp_dir in os.listdir(RESULT_DIR) if "tmp_" in tmp_dir]:
    shutil.rmtree(f"{RESULT_DIR}/{tmp_dir}")

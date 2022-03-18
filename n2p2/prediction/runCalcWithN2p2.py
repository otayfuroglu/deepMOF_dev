#!/bin/env python

from __future__ import absolute_import
from __future__ import print_function
import pynnp
import numpy as np
from ase.db import connect
import os, sys
import torch


from n2p2AseInterFace import n2p2Calculator
import argparse

parser = argparse.ArgumentParser(description="Give something ...")
#  parser.add_argument("-mof_num", "--mof_num",
#                      type=int, required=True,
                    #  help="..")
parser.add_argument("-val_type", "--val_type",
                    type=str, required=True,
                    help="..")
parser.add_argument("-data_path", "--data_path",
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
args = parser.parse_args()


def aseDB2n2p2(row):
    fl = open("input.data", "w")
    atoms_prep_list = [["begin"], ["comment ", row.name]]
    atom_template = 'atom {:10.6f} {:10.6f} {:10.6f} {:2s} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}'
    atoms_prep_list += [[atom_template.format(
        position[0], position[1], position[2],
        symbol, 0.0, 0.0,
        forces[0], forces[1], forces[2])]
        for position, symbol, forces in zip(
            row.positions.tolist(), row.symbols, row.forces.tolist())]
    atoms_prep_list += [["energy ", row.energy], ["charge 0.0"], ["end"]]
    for line in atoms_prep_list:
        for item in line:
            fl.write(str(item))
        fl.write("\n")
    fl.close()


def prepareCalc(best_epoch):

    os.system(f"cp {MODEL_DIR}/input.nn {RESULT_DIR}/")
    os.system(f"cp {MODEL_DIR}/scaling.data {RESULT_DIR}/")

    weights_files = [item for item in os.listdir(MODEL_DIR) if "weights" in item]
    best_weights_files = [item for item in weights_files if int(item.split(".")[-2]) == best_epoch]
    assert len(best_weights_files) != 0, "Erro: NOT FOUND best epoch number"

    for best_weights_file in best_weights_files:
        os.system(f"cp {MODEL_DIR}/{best_weights_file} {RESULT_DIR}/{best_weights_file[:11]}.data")


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


def runPredict(row, fl_obj):
    aseDB2n2p2(row)
    # Initialize NNP prediction mode.
    p = pynnp.Prediction()
    # Read settings and setup NNP.
    p.setup()
    # Read in structure.
    p.readStructureFromFile()
    # Predict energies and forces.
    p.predict()
    # Shortcut for structure container.
    s = p.structure

    natoms = len(row.symbols)
    n2p2_forces = np.zeros([natoms,3])
    for i, atom in enumerate(s.atoms):
        n2p2_forces[i, :] = atom.f.r

    # first get fmax indices than get qm_fmax and n2p2 fmax component
    qm_fmax_component_idx = get_fmax_idx(row.forces)
    qm_fmax_component = get_fmax_componentFrom_idx(row.forces,
                                                   qm_fmax_component_idx)
    n2p2_fmax_component = get_fmax_componentFrom_idx(n2p2_forces, qm_fmax_component_idx)

    diffE = s.energyRef - s.energy
    diffEpa = diffE / natoms

    diffFamxComp = n2p2_fmax_component - qm_fmax_component
    print("{},{},{},{}".format(row.name, diffE, diffEpa, diffFamxComp), file=fl_obj)
   #   print("------------")
   #   print("Structure 1:")
   #   print("------------")
   #   print("numAtoms           : ", s.numAtoms)
   #   print("numAtomsPerElement : ", s.numAtomsPerElement)
   #   print("------------")
   #   print("Energy (Ref) : ", s.energyRef)
   #   print("Energy (NNP) : ", s.energy)
   #   print("------------")
   #   for atom in s.atoms:
   #       print(atom.index, atom.element, p.elementMap[atom.element], atom.f.r)

def runPredict_test(row, calculator, fl_obj):

    atoms = row.toatoms()
    atoms.set_calculator(calculator)

    qm_energy = row.energy
    # first get fmax indices than get qm_fmax and n2p2 fmax component
    qm_fmax_component_idx = get_fmax_idx(row.forces)
    qm_fmax_component = get_fmax_componentFrom_idx(row.forces,
                                                   qm_fmax_component_idx)

    n2p2_energy = atoms.get_potential_energy()
    n2p2_forces = atoms.get_forces()
    n2p2_fmax_component = get_fmax_componentFrom_idx(n2p2_forces, qm_fmax_component_idx)

    diffE = qm_energy - n2p2_energy
    diffEpa = diffE / len(atoms)

    diffFamxComp = n2p2_fmax_component - qm_fmax_component
    print("{},{},{},{}".format(row.name, diffE, diffEpa, diffFamxComp), file=fl_obj)
   #   print("------------")
   #   print("Structure 1:")
   #   print("------------")
   #   print("numAtoms           : ", s.numAtoms)
   #   print("numAtomsPerElement : ", s.numAtomsPerElement)
   #   print("------------")
   #   print("Energy (Ref) : ", s.energyRef)
   #   print("Energy (NNP) : ", s.energy)
   #   print("------------")
   #   for atom in s.atoms:
   #       print(atom.index, atom.element, p.elementMap[atom.element], atom.f.r)

mode = args.val_type
MODEL_DIR = args.MODEL_DIR
RESULT_DIR = args.RESULT_DIR
data_path = args.data_path


def main():

    #  prepareCalc(best_epoch=57)
    #  os.chdir(RESULT_DIR)
    #  db_path = "../../../../deepMOF/HDNNP/prepare_data/workingOnDataBase/"\
    #      + "nonEquGeometriesEnergyForcesWithORCAFromMD_testSet.db"
        #  + "nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling.db" # for  just MOF5
        #  + "nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling_IRMOFseries1_4_6_7_10_merged_50000_ev.db"
    os.chdir(RESULT_DIR)
    calculator = n2p2Calculator(model_dir=MODEL_DIR, best_epoch=args.best_epoch)
    db = connect(data_path)
    db = db.select()
    fl_obj = open("results.csv", "w")
    fl_obj.write("Name,DifEnergy(eV),DifEnergyPa(eV),diffFmaxComp(eV/A)\n")
    for i, row in enumerate(db):
        runPredict_test(row, calculator, fl_obj)
        if i == 30:
            break
    fl_obj.close()

main()

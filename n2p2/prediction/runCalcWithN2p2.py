#!/bin/env python

from __future__ import absolute_import
from __future__ import print_function
import pynnp
from ase.db import connect
import os, sys


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

    print("{},{}".format(row.name, (s.energyRef-s.energy)), file=fl_obj)
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

    prepareCalc(best_epoch=7)
    os.chdir(RESULT_DIR)
    #  db_path = "../../../../deepMOF/HDNNP/prepare_data/workingOnDataBase/"\
    #      + "nonEquGeometriesEnergyForcesWithORCAFromMD_testSet.db"
        #  + "nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling.db" # for  just MOF5
        #  + "nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling_IRMOFseries1_4_6_7_10_merged_50000_ev.db"
    db = connect(data_path)
    db = db.select()
    fl_obj = open("results.csv", "w")
    fl_obj.write("Name,DifEnergy(eV)")
    for i, row in enumerate( db ):
        runPredict(row, fl_obj)
        if i == 100:
            break

main()

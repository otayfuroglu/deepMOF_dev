
from ase import units
from ase.db import connect
import numpy as np

import tqdm
import argparse
import os
#  import shutil


u = units.create_units("2014")
EV2HARTREE = 1.0 / u["Hartree"]
ANG2BOHR = 1.0 / u["Bohr"]

def aseDb2Runner(db):
    db = db.select()
    fl = open(f"input.data", "w")
    for i, row in enumerate(tqdm.tqdm(db)):
        # remove random value for efficiency
        atoms_prep_list = [["begin"]]
        atoms_prep_list += [["lattice    {:15.8f}    {:15.8f}    {:15.8f}".format(
                cell[0], cell[1], cell[2])] for cell in row.cell * ANG2BOHR]

        atom_template = 'atom {:15.8f} {:15.8f} {:15.8f} {:2s} {:15.8f} {:15.8f} {:15.8f} {:15.8f} {:15.8f}'
        atoms_prep_list += [[atom_template.format(
            position[0], position[1], position[2],
            symbol, charge, 0.0,
            forces[0], forces[1], forces[2])]
            for position, symbol, charge, forces in zip(
                (row.positions * ANG2BOHR).tolist(),
                row.symbols,
                row.data.charges,
                #  chgarges_list,
                (np.array(row.data.forces) * (EV2HARTREE/ANG2BOHR)).tolist())]

        atoms_prep_list += [["energy ",
                             row.data.energy * EV2HARTREE], ["charge 0.0"], ["end"]]

        for line in atoms_prep_list:
            for item in line:
                fl.write(str(item))
            fl.write("\n")
    fl.close()

def rndAseDb2Runner(db):
    import random
    rand_list = random.sample(range(db.count()), N)
    db = db.select()
    fl = open(f"input.data.rand{N}", "w")
    for i, row in enumerate(tqdm.tqdm(db)):
        if i in rand_list:
            # remove random value for efficiency
            rand_list.remove(i)
            atoms_prep_list = [["begin"]]
            atoms_prep_list += [["lattice    {:15.8f}    {:15.8f}    {:15.8f}".format(
                    cell[0], cell[1], cell[2])] for cell in row.cell * ANG2BOHR]

            atom_template = 'atom {:15.8f} {:15.8f} {:15.8f} {:2s} {:15.8f} {:15.8f} {:15.8f} {:15.8f} {:15.8f}'
            atoms_prep_list += [[atom_template.format(
                position[0], position[1], position[2],
                symbol, charge, 0.0,
                forces[0], forces[1], forces[2])]
                for position, symbol, charge, forces in zip(
                    (np.array(row.positions) * ANG2BOHR).tolist(),
                    row.symbols,
                    row.data.charges,
                    #  chgarges_list,
                    (np.array(row.data.forces) * (EV2HARTREE/ANG2BOHR)).tolist())]

            atoms_prep_list += [["energy ",
                                 row.data.energy * EV2HARTREE], ["charge 0.0"], ["end"]]

            for line in atoms_prep_list:
                for item in line:
                    fl.write(str(item))
                fl.write("\n")
    fl.close()

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-db", type=str, required=True)
parser.add_argument("-N", type=int, default=0, required=False)
args = parser.parse_args()


db = connect(args.db)
N = args.N

if N == 0:
    aseDb2Runner(db)
else:
    rndAseDb2Runner(db)


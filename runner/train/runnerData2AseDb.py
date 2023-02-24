
from ase.db import connect
from ase import units
import numpy as np
from ase import Atoms

from ase.db import connect
from ase.db.core import Database
import argparse
import os



u = units.create_units("2014")
HARTREE2EV = u["Hartree"]
BOHR2ANG = u["Bohr"]


parser = argparse.ArgumentParser(description="")
parser.add_argument("-data", type=str, required=True)
args = parser.parse_args()

runner_data = args.data
ase_db = f"{runner_data.split('.')[0]}.db"

if os.path.exists(ase_db):
    os.remove(ase_db)
db = connect(ase_db)

lattice = []
positions = []
symbols = []
charges = []
forces = []

with open("input.data.300") as lines:
    for line in lines:
        if "lattice" in line:
            lattice += [[float(item)*BOHR2ANG for item in line.split()[1:]]]
        if "atom" in line:
            positions += [[float(item)*BOHR2ANG for item in line.split()[1:4]]]
            symbols += [line.split()[4]]
            charges += [float(item) for item in line.split()[5:6]]
            forces += [[float(item)*(HARTREE2EV/BOHR2ANG) for item in line.split()[7:10]]]
        if "energy" in line:
            energy = float(line.split()[-1])
        if "end" in line:
            atoms = Atoms(symbols=symbols, positions=positions, cell=lattice)
            data = {}

            data["energy"] = energy * HARTREE2EV
            data["forces"] = forces
            data["charges"] = charges
            db.write(atoms=atoms, data=data)

            # clerr list
            lattice = []
            positions = []
            symbols = []
            charges = []
            forces = []

print(f"ASE Database is created:")
print(f"Numer of image: {db.count()}")


import os
from coordNum import coordnum
from ase import Atoms
from ase.io import read, write
from ase.geometry import cell_to_cellpar, cellpar_to_cell

from tqdm import tqdm

import argparse


def atoms2Ascii(atoms):

    # number of atoms
    atoms_prep_list = [[f"{len(atoms):7.0f}"]]

    # cell parameters
    cell = cellpar_to_cell(cell_to_cellpar(atoms.cell))
    dxx = cell[0, 0]
    dyx, dyy = cell[1, 0:2]
    dzx, dzy, dzz = cell[2, 0:3]

    cell_template = '{:15.12f} {:15.12f} {:15.12f}'
    atoms_prep_list += [cell_template.format(dxx, dyx, dyy)]
    atoms_prep_list += [cell_template.format(dzx, dzy, dzz)]

    # positons and symbols
    atom_template = '{:15.12f} {:15.12f} {:15.12f} {:2s}'
    atoms_prep_list += [[atom_template.format(position[0], position[1], position[2], symbol)]
            for position, symbol, in zip( (atoms.positions).tolist(), atoms.symbols)]

    with open(f"tmp.ascii", "w") as fl:
        for line in atoms_prep_list:
            for item in line:
                fl.write(str(item))
            fl.write("\n")


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-trj_path", type=str, required=True, help="..")
parser.add_argument("-interval", type=int, required=False, default=1, help="..")
#  parser.add_argument("-nproc", type=int, required=True, help="..")
args = parser.parse_args()
trj_path = args.trj_path

idxs = slice(0, -1, args.interval)
atoms_list = read(trj_path, index=idxs)


fl_out = f"with_coordn_{trj_path.split('/')[-1]}"
if os.path.exists(fl_out):
    os.remove(fl_out)

fl = open("av_coordnums.csv", "w")
fl.write("index,AvCoordnum\n")
for i, atoms in enumerate(tqdm(atoms_list)):
    atoms2Ascii(atoms)
    nat = len(atoms)
    #  print(nat)

    coordnum("tmp.ascii", "tmp.extxyz", nat, atom_symb="Al")
    atoms = read("tmp.extxyz")
    coordnums = atoms.arrays["coordn"]
    av_coordnum = coordnums[coordnums > 0.0].mean()
    fl.write(f"{i},{av_coordnum}\n")
    os.remove("tmp.ascii")
    os.remove("tmp.extxyz")
    #  coordnum = coordnum("tmp.ascii", 12)
    #  break

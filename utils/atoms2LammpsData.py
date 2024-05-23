
from ase.io import read, write
from ase.io.lammpsdata import write_lammps_data
import argparse

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-flpath", type=str, required=True, help="..")
parser.add_argument("-specorder", type=list, nargs='+', required=True, help="..")
args = parser.parse_args()

flpath = args.flpath
atoms = read(flpath)
#  atoms.center(vacuum=5.0)
#  write("./MgF1.extxyz", atoms)
write_lammps_data(f"data.{flpath.split('/')[-1].split('.')[0]}",
                  atoms,
                  specorder=["Mg", "O", "C", "H"]
                 )

#
from ase import Atoms
from ase.io import read, write
import argparse
import os
import json

def lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair):
    symbols = [atom_type_symbol_pair[str(key)] for key in lammps_atoms.get_atomic_numbers()]
    return Atoms(symbols=symbols, positions=lammps_atoms.positions, cell=lammps_atoms.cell)


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-trj_path", type=str, required=True, help="..")
parser.add_argument("-atom_type_symbol_pair", type=str, required=True, help="give atoms number and symbols accordingly lammps type like dictionary but enclosed in double quotes both of keys and values")
parser.add_argument("-out_format", type=str, required=True, help="..")
# eg. -atom_type_symbol_pair '{"1":"C", "2":"O", "3":"H", "4":"O", "5":"Mg"'
parser.add_argument("-interval", type=int, required=False, default=1, help="..")
args = parser.parse_args()

idxes = slice(0, -1, args.interval)
atom_type_symbol_pair = args.atom_type_symbol_pair
atom_type_symbol_pair = json.loads(args.atom_type_symbol_pair)
lammps_trj = read(args.trj_path, format="lammps-dump-text", index=idxes, parallel=True)
atoms = [lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair) for lammps_atoms in lammps_trj]
write(f"{os.path.splitext(args.trj_path)[0]}.{args.out_format}", atoms)

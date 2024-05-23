#
from ase import Atoms
from ase.io import read, write
import argparse
import os

def lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair):
    symbols = [atom_type_symbol_pair[key] for key in lammps_atoms.get_atomic_numbers()]
    return Atoms(symbols=symbols, positions=lammps_atoms.positions, cell=lammps_atoms.cell)


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-trj_path", type=str, required=True, help="..")
parser.add_argument("-out_format", type=str, required=True, help="..")
parser.add_argument("-interval", type=int, required=False, default=1, help="..")
args = parser.parse_args()

idxes = slice(0, -1, args.interval)
#  atom_type_symbol_pair = {1:"Al", 2:"Li", 3:"H"}

# for MgF1_charged
#  atom_type_symbol_pair = {
#      1: "C",
#      2: "O",
#      3: "O",
#      4: "Mg",
#      5: "O",
#      6: "H",
#      7: "C",
#      8: "C",
#      9: "O",
#  }

# for MgF1_charged_bigcell
atom_type_symbol_pair = {
    1: "C",
    2: "O",
    3: "O",
    4: "Mg",
    5: "C",
    6: "O",
    7: "H",
    8: "C",
    9: "O",
}
#  lammps_trj_path = "../alanates/cscs/nnp_train_on16kdata_nvt_02timestep_1500K_2ns/alanates_1Bar_1500K.lammpstrj"
lammps_trj = read(args.trj_path, format="lammps-dump-text", index=idxes, parallel=True)
atoms = [lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair) for lammps_atoms in lammps_trj]
write(f"{os.path.splitext(args.trj_path)[0]}.{args.out_format}", atoms)

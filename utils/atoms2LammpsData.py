
from ase.io import read, write
from ase.io.lammpsdata import write_lammps_data


flpath = "./mg_mof74.extxyz"
atoms = read(flpath)
#  atoms.center(vacuum=5.0)
#  write("./MgF1.extxyz", atoms)
write_lammps_data(f"data.{flpath.split('/')[-1].split('.')[0]}",
                  atoms,
                  specorder=["Mg", "O", "C", "H"]
                 )

#
from ase import Atoms
from ase.cell import Cell
from ase.io import read, write
from lammps import lammps
import numpy as np
import argparse



def atoms2lammpsdata(atoms):
    # Write CO2 data with velocities to LAMMPS-compatible file
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.io.lammpsdata import write_lammps_data

    if np.all(atoms.get_velocities()==0):
        MaxwellBoltzmannDistribution(atoms, 100)

    #  print(atoms.cell)
    data_file = f"data.{file_base}"
    #  write_lammps_data_with_velocities(data_file, atoms)
    write_lammps_data(data_file, atoms, units="metal", specorder=specorder, velocities=False)
    #  write("test_trj.extxyz", atoms, append=True)
    #NOTE: there is some problem velocity unit convertsioin with write_lammps_data.
    # we write manual below 
    with open(data_file, 'a') as f:
        f.write("\n\nVelocities\n\n")
        for i, vel in enumerate(atoms.get_velocities(), start=1):
            f.write(f"{i} {vel[0]:.6f} {vel[1]:.6f} {vel[2]:.6f}\n")


def run(nsteps):
    # Initialize LAMMPS
    lmp = lammps()
    lmp.command("clear")
    lmp.command("units metal")
    lmp.command("dimension 3")
    lmp.command("boundary p p p")
    lmp.command("atom_style atomic")
    lmp.command("newton off")
    lmp.command(f"read_data data.{file_base}")
    lmp.command(f"replicate 1 2 2")

    for i, symbol in enumerate(specorder):
        lmp.command(f"mass {i+1} {masses[symbol]}")

    lmp.command("pair_style nequip")
    lmp.command(f"pair_coeff * * {model_path} {' '.join(specorder)}")
    lmp.command("thermo 50")
    lmp.command("compute moltemp all temp")
    lmp.command("compute_modify moltemp dynamic/dof yes")
    lmp.command("compute_modify thermo_temp dynamic/dof yes")
    lmp.command("neighbor 2.0 bin")
    lmp.command("neigh_modify every 1 delay 10 check yes")
    lmp.command(f"velocity all create {temperature} 54654")
    lmp.command(f"timestep {timestep}")
    lmp.command(f"minimize 1.0e-15 1.0e-15 1000 10000")
    lmp.command(f"fix mynvt all nvt temp {temperature} {temperature} 0.5")
    lmp.command(f"run 50000")
    lmp.command(f"unfix mynvt")
    lmp.command(f"reset_timestep 0")
    lmp.command(f"thermo_style custom step temp press pe ke density atoms vol")
    lmp.command(f"log {file_base}_{pressure}bar_{temperature}K.log")
    lmp.command(f"dump 1 all atom 50 {file_base}_{pressure}bar_{temperature}K.lammpstrj")
    #  lmp.command(f"fix mynpt all npt temp {temperature} {temperature} 0.1 iso {pressure} {pressure} 0.5")
    lmp.command(f"fix mynpt all npt temp {temperature} {temperature} 0.1 tri {pressure} {pressure} 0.5")
    #  lmp.command(f"run {nsteps}")


    lmp.command(f"run {nsteps}")
    # Finalize LAMMPS
    lmp.close()



parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-file_base", type=str, required=True)
parser.add_argument("-temp", type=int, required=True)
args = parser.parse_args()


file_base = args.file_base
masses = {
    "Al": 26.981,
    "Li": 6.941,
    "H": 1.00794,
}

model_path = "alanatesWithStress.pth"
#  specorder=["Mn", "O", "C", "H"]
specorder = list(masses.keys())
#  atom_type_pairs = {"Mg": 1, "O": 2,  "C": 3, "H": 4}
temperature = args.temp  # Kelvin
pressure = 0
timestep = 0.0005  # ps
nsteps = 4000000

#  atoms = read(f"{file_base}.cif")
atoms = read(f"{file_base}.extxyz")
atoms2lammpsdata(atoms)

run(nsteps)

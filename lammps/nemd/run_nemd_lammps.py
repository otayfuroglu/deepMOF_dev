#

#  NEMD for binary gas system in a MOF using the LAMMPS Python API.
#
#  - System: MOF framework + GAS1 + GAS2
#  - Method: Apply a constant body force along z to gas species (NEMD)
#  - Output: density profiles along z for each component (separation analysis)
#
#  Adapt atom types, force field, and file names to your system.
#
from ase.io import read
from lammps import lammps
import numpy as np
import os
import argparse



def run_lammps_nemd():
    # Initialize LAMMPS
    lmp = lammps()  # you can pass cmdargs if needed

    # Helper: send a multi-line LAMMPS input string
    def lmp_block(cmd_block: str):
        for line in cmd_block.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                lmp.command(line)

    # ------------------------------
    # LAMMPS setup: units, atom style, read data
    # ------------------------------

    lmp.command("clear")
    lmp.command("units metal")
    lmp.command("dimension 3")
    lmp.command("boundary p p p")
    lmp.command("atom_style full")
    lmp.command("newton off")
    lmp.command(f"read_data data.{file_base}")
    lmp.command(f"replicate 1 1 1")

    for key, values in atom_type_pairs_frame.items():
        lmp.command(f"mass {key} {values[1]}")
    for key, values in atom_type_pairs_gas1.items():
        lmp.command(f"mass {key} {values[1]}")
    for key, values in atom_type_pairs_gas2.items():
        lmp.command(f"mass {key} {values[1]}")

    # ------------------------------
    # Group definitions
    # ------------------------------
    lmp_block(f"""
    group mof   type {MOF_TYPES}
    group graphene   type {GRAPHENE_TYPES}
    group co2   type {GAS1_TYPES}
    group ch4   type {GAS2_TYPES}
    group gas   union co2 ch4
    group system   union gas mof
    """)

    if sim_type == "rigid":
        lmp_block(f"""
        #  Freeze MOF framework
        velocity mof set 0.0 0.0 0.0
        fix freeze_mof mof setforce 0.0 0.0 0.0
        """)

    # ------------------------------
    # Force field (placeholder – ADAPT THIS)
    # ------------------------------
    # You should replace this with your own FF (TraPPE for GAS2, EPM2 for GAS1, MOF FF, etc.)
    # Just to make the template self-contained, I use a generic LJ + Coulomb.
    lmp_block(f"""
    neighbor        2.0 bin
    neigh_modify    every 1 delay 0 check yes

    pair_style      nequip
    pair_coeff * * {model_path} {' '.join(specorder)}

    """)


    # Log
    lmp.command(f"log {file_base}_{TEMP}K.log")
    # ------------------------------
    # Thermo output
    # ------------------------------
    lmp_block(f"""
    thermo          {THERMO_EVERY}
    thermo_style    custom step temp press pe ke etotal density
    """)

    # ------------------------------
    # Temperature control for gas phase (NVT)
    # ------------------------------
    #  lmp.command(f"velocity gas create 200 54654")

    lmp.command(f"timestep        {TIMESTEP} ")

    if sim_type == "rigid":
        lmp_block(f"""
        # Equilibration in NVT (only gas moves)
        fix nvt_eq system nvt temp {TEMP} {TEMP} 0.1
        """)
    elif sim_type == "flex":
        lmp_block(f"""
        # Equilibration in NVT (system, mof + gas, moves)
        fix nvt_eq system nvt temp {TEMP} {TEMP} 0.1
        """)

    #  lmp_block(f"""
    #
    #  # Optional: remove net rotation and drift of gas (not along driven direction)
    #  # Be cautious with 'momentum' if you are driving along z. Here we only
    #  # remove drift in x and y.
    #
    #  fix mom_removal gas momentum 100 linear 1 1 0
    #  """)

    # ------------------------------
    # Dump for visualization / analysis
    # ------------------------------
    lmp_block(f"""
    dump dump_all all custom {DUMP_EVERY} {file_base}_{TEMP}K.lammpstrj id type x y z vx vy vz
    dump_modify dump_all sort id
    """)

    # optimizaton run
    lmp.command("min_style cg")
    #  lmp.command("minimize 1.0e-8 1.0e-4 1000 10000")
    lmp.command("minimize 1.0e-4 1.0e-2 1000 10000")

    # ------------------------------
    # Equilibration run
    # ------------------------------
    print(">>> Equilibration (NVT)...")
    lmp.command(f"run {EQUIL_STEPS}")

    # ------------------------------
    # Switch to NEMD: keep thermostat, add driving force
    # ------------------------------
    # Define force magnitudes as variables
    lmp_block(f"""
    unfix nvt_eq
    #  unfix mom_removal
    """)

    if sim_type == "rigid":
        lmp_block(f"""
        # New NVT with same T (now for NEMD)
        fix nvt_prod gas nvt temp {TEMP} {TEMP} 0.1
        """)

    if sim_type == "flex":
        lmp_block(f"""
        # New NVT with same T (now for NEMD)
        fix nvt_prod system nvt temp {TEMP} {TEMP} 0.1
        """)

    #  lmp_block(f"""
    #  fix mom_xy gas momentum 100 linear 1 1 0
    #
    #  variable Fco2z equal {F_GAS1_Z}
    #  variable Fch4z equal {F_GAS2_Z}
    #
    #  # Apply body force along +z for each component
    #  fix drift_gas1 co2 addforce 0.0 0.0 v_Fco2z
    #  fix drift_gas2 ch4 addforce 0.0 0.0 v_Fch4z
    #  """)

    # ------------------------------
    # Spatial binning along z for density profiles
    # ------------------------------
    # We use chunk/atom bin/1d z lower Δz and average densities in each bin.
    # The bin width (0.5 Å here) can be changed.
    #  lmp_block(f"""
    #  # Define bins along z
    #  compute zchunk all chunk/atom bin/1d z lower 0.5 units box
    #
    #  # Density profile for all atoms
    #  fix prof_all all ave/chunk 100 100 10000 zchunk density/mass file density_profile_{file_base}_{TEMP}K.dat ave running
    #
    #  # Optionally, separate profiles per species:
    #  compute zchunk_gas1 co2 chunk/atom bin/1d z lower 0.5 units box
    #  fix prof_gas1 co2 ave/chunk 100 100 10000 zchunk_gas1 density/mass file density_profile_gas1_{file_base}_{TEMP}K.dat ave running
    #
    #  compute zchunk_gas2 ch4 chunk/atom bin/1d z lower 0.5 units box
    #  fix prof_gas2 ch4 ave/chunk 100 100 10000 zchunk_gas2 density/mass file density_profile_gas2_{file_base}_{TEMP}K.dat ave running
    #  """)

    # ------------------------------
    # NEMD production run
    # ------------------------------
    print(">>> NEMD production with driving force...")
    lmp.command(f"run {NEMD_STEPS}")

    print(">>> NEMD run complete.")
    lmp.close()



parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-file_base", type=str, required=True)
parser.add_argument("-model_path", type=str, required=True)
parser.add_argument("-sim_type", type=str, required=True)
parser.add_argument("-temp", type=int, required=True)
args = parser.parse_args()



# ------------------------------
# User parameters (edit these)
# ------------------------------

#  DATA_FILE = "mof_gas1_gas2.data"   # your LAMMPS data file
#  LOG_FILE = "log.nemd_mof_gas1_gas2.lammps"
#  DUMP_FILE = "dump.nemd_mof_gas1_gas2.lammpstrj"
#  PROFILE_FILE = "density_profile_nemd.dat"

# Simulation parameters
TEMP = args.temp       # Kelvin
TIMESTEP = 0.0005         # ps (units metal)
#  EQUIL_STEPS = 200000   # NVT equilibration
EQUIL_STEPS = 50000   # NVT equilibration
#  NEMD_STEPS = 500000    # production NEMD
NEMD_STEPS = 500000    # production NEMD
THERMO_EVERY = 50
DUMP_EVERY = 50

# External driving force along z ( eV/Å)
F_GAS1_Z = 0.002168
F_GAS2_Z = 0.002168
#  F_GAS1_Z = 0.001
#  F_GAS2_Z = 0.001

atom_type_pairs_frame = {1: ["Mg", 24.3050], 2: ["O", 15.9994],  3: ["C", 12.0107], 4: ["H", 1.00794]}
atom_type_pairs_graphene = {5: ["C", 12.0107]}
atom_type_pairs_gas1 = {6: ["C", 12.0107], 7: ["O", 15.9994]}
#  atom_type_pairs_gas2 = {8: ["C", 12.0107], 9: ["H", 1.00794]}
atom_type_pairs_gas2 = {8: ["H", 1.00794]}

specorder = [value[0] for value in list(atom_type_pairs_frame.values())]\
                + [value[0] for value in list(atom_type_pairs_graphene.values())
                                              + list(atom_type_pairs_gas1.values())
                                              + list(atom_type_pairs_gas2.values())]

# Atom type mapping (adapt to your data file!)
MOF_TYPES = " ".join(list(str(i) for i in atom_type_pairs_frame.keys()))      # framework atom types 
GRAPHENE_TYPES = " ".join(list(str(i) for i in atom_type_pairs_graphene.keys()))  # graphene atom types
GAS1_TYPES = " ".join(list(str(i) for i in atom_type_pairs_gas1.keys()))    # C and O atoms of GAS1 (example)
GAS2_TYPES = " ".join(list(str(i) for i in atom_type_pairs_gas2.keys()))       # GAS2 

file_base = args.file_base
model_path = args.model_path
sim_type = args.sim_type.lower()

run_lammps_nemd()


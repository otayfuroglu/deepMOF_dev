#
from ase.io import read
from lammps import lammps
import numpy as np
import os, sys
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
    for key, values in atom_type_pairs_gas.items():
        lmp.command(f"mass {key} {values[1]}")

    # ------------------------------
    # Group definitions
    # ------------------------------
    lmp_block(f"""
    group mof   type {MOF_TYPES}
    group gas   type {GAS_TYPES}
    group system   union gas mof
    """)

    lmp_block(f"""
    neighbor        2.0 bin
    neigh_modify    every 1 delay 0 check yes

    pair_style      nequip
    pair_coeff * * {model_path} {' '.join(specorder)}
    """)

    # Log
    lmp.command(f"log {file_base}_{TEMP}K_eq.log")

    # ------------------------------
    # Thermo output
    # ------------------------------
    lmp_block(f"""
    thermo          {THERMO_EVERY}
    thermo_style    custom step temp press pe ke etotal vol
    """)


    lmp_block("""
    compute Tgas gas temp
    thermo_modify temp Tgas
    """)

    # ------------------------------
    # Timestep
    # ------------------------------
    lmp.command(f"timestep {TIMESTEP}")


    lmp.command(f"velocity gas create {TEMP} 54654 mom yes rot yes dist gaussian")

    # ============================================================
    # EQUILIBRATION
    # ============================================================
    print(">>> Equilibration ...")

    if sim_type == "rigid":
        # ---- RIGID GAS: integrate with rigid/small + thermostat with langevin
        # ---- MOF frozen (only gas moves)
        lmp_block(f"""
        # Freeze MOF atoms
        fix freeze mof setforce 0.0 0.0 0.0
        velocity mof set 0.0 0.0 0.0

        # Nose–Hoover NVT thermostat for rigid gas bodies (integrator + thermostat)
        fix riggas gas rigid/nvt/small molecule temp {TEMP} {TEMP} {100*TIMESTEP}

        #  # Rigid gases
        #  fix rig gas rigid/small molecule
        #  # Thermostat rigid gas
        #  fix tgas gas langevin {TEMP} {TEMP} 0.1 12345


        # (Optional) remove any drift in gas COM
        #  fix mom gas momentum 100 linear 1 1 1
        """)

    elif sim_type == "flex":
        lmp_block(f"""
        # ---- FLEX: standard NVT on whole system
        #  fix npt_eq system npt temp {TEMP} {TEMP} 0.1 iso {PRES} {PRES} 0.5

        # gas rigid in NVT (often preferred even if MOF is NPT)
        #  fix riggas gas rigid/nvt/small molecule temp {TEMP} {TEMP} {100*TIMESTEP}

        # MOF in  NVT
        #  fix flexmof mof nvt temp {TEMP} {TEMP} {100*TIMESTEP}

        compute mdtemp gas temp
        compute_modify  mdtemp dynamic/dof yes
        fix riggas gas rigid/small molecule
        fix_modify riggas dynamic/dof yes

        # MOF in NPT
        fix flexmof mof npt temp {TEMP} {TEMP} 100 iso 1.0 1.0 {1000*TIMESTEP}

        """)

    # optimizaton run with cell
    #  lmp.command("fix boxrelax all box/relax iso 0.0 vmax 0.001")
    # optimizaton
    #  lmp.command("min_style cg")
    #  lmp.command("minimize 1.0e-8 1.0e-4 1000 10000")

    # Run equilibration
    lmp.command(f"run {EQUIL_STEPS}")

    # Clean up equilibration fixes
    #  if sim_type == "rigid":
    #      lmp_block("""
    #      #  unfix mom
    #      unfix tgas
    #      unfix rig
    #      unfix freeze
    #      """)
    #  elif sim_type == "flex":
    #      lmp_block("""
    #      unfix npt_eq
    #      """)

    # ============================================================
    # PRODUCTION
    # ============================================================
    print(">>> Production ...")

    # reset time
    lmp.command(f"reset_timestep 0")
    lmp.command(f"log {file_base}_{TEMP}K_prod.log")
    lmp.command(f"restart 500000 {file_base}_{TEMP}K.restart")
    # ------------------------------
    # Dump for visualization / analysis
    # ------------------------------
    lmp_block(f"""
    dump dump_all all custom {DUMP_EVERY} {file_base}_{TEMP}K.lammpstrj id mol type xu yu zu
    dump_modify dump_all sort id
    """)

    #  if sim_type == "rigid":
    #      # Re-apply rigid + thermostat for production (same as equil)
    #      lmp_block(f"""
    #      # Freeze MOF atoms
    #      fix freeze mof setforce 0.0 0.0 0.0
    #      velocity mof set 0.0 0.0 0.0
    #
    #      # Rigid gases
    #      fix rig gas rigid/small molecule
    #
    #      # Thermostat rigid bodies
    #      fix tgas gas nvt temp {TEMP} {TEMP} 0.1
    #      """)
    #
    #  elif sim_type == "flex":
    #      lmp_block(f"""
    #      fix npt_prod system npt temp {TEMP} {TEMP} 0.1 iso {PRES} {PRES} 0.5
    #      """)

    # ------------------------------
    # MSD computation (gas)
    # ------------------------------
    lmp_block(f"""
    # Chunk molecules (requires molecule IDs in the data file)
    compute ch_gas gas chunk/atom molecule ids once

    # Per-molecule MSD (LAMMPS msd/chunk output: dx^2 dy^2 dz^2 total)
    compute msdgas gas msd/chunk ch_gas

    # Write raw per-molecule MSD (block format, what you already have)
    fix outgas gas ave/time 100 1 100 c_msdgas[*] file msd_gas.data mode vector

    """)

    # Run production
    lmp.command(f"run {NEMD_STEPS}")

    #  # Clean up production fixes
    #  if sim_type == "rigid":
    #      lmp_block("""
    #      unfix outgas
    #      unfix tgas
    #      unfix rig
    #      unfix freeze
    #      """)
    #  elif sim_type == "flex":
    #      lmp_block("""
    #      unfix outgas
    #      unfix nvt_prod
    #      """)

    print(">>> Run complete.")
    lmp.close()




parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-file_base", type=str, required=True)
parser.add_argument("-model_path", type=str, required=True)
parser.add_argument("-sim_type", type=str, required=True)
parser.add_argument("-gas_type", type=str, required=True)
parser.add_argument("-temp", type=int, required=True)
args = parser.parse_args()


file_base = args.file_base
model_path = args.model_path
sim_type = args.sim_type.lower()
gas_type = args.gas_type.lower()


# ------------------------------
# User parameters (edit these)
# ------------------------------

#  DATA_FILE = "mof_co2_ch4.data"   # your LAMMPS data file
#  LOG_FILE = "log.nemd_mof_co2_ch4.lammps"
#  DUMP_FILE = "dump.nemd_mof_co2_ch4.lammpstrj"
#  PROFILE_FILE = "density_profile_nemd.dat"

# Simulation parameters
TEMP = args.temp       # Kelvin
PRES = 1.0
TIMESTEP = 0.001         # ps (units metal)

EQUIL_STEPS = 20000   # NVT equilibration
if sim_type == "rigid":
    EQUIL_STEPS = 10000   # NVT equilibration
NEMD_STEPS = 2000000    # production MD

THERMO_EVERY = 50
DUMP_EVERY = 20


atom_type_pairs_frame = {1: ["Mg", 24.3050], 2: ["O", 15.9994],  3: ["C", 12.0107], 4: ["H", 1.00794]}
if gas_type == "co2":
    atom_type_pairs_gas = {5: ["C", 12.0107], 6: ["O", 15.9994]} # for CO2
elif gas_type == "ch4":
    atom_type_pairs_gas = {5: ["C", 12.0107], 6: ["H", 1.00794]} # for CH4
else:
    print("Please enter gas type as 'CH4' or 'CO2'")
    sys.exit(1)

specorder = [value[0] for value in list(atom_type_pairs_frame.values())]\
                + [value[0] for value in list(atom_type_pairs_gas.values())]

# Atom type mapping (adapt to your data file!)
MOF_TYPES = " ".join(list(str(i) for i in atom_type_pairs_frame.keys()))      # framework atom types 
GAS_TYPES = " ".join(list(str(i) for i in atom_type_pairs_gas.keys()))    # C and O atoms of CO2 (example)

run_lammps_nemd()


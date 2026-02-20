#

#  EMD for binary gas system in a MOF using the LAMMPS Python API.
#
#  - System: MOF framework + CO2 + CH4
#  - Method: Apply a constant body force along z to gas species (EMD)
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


    # ==========================================
    # Binary gas (gas1 + gas2) EMD block (generic)
    # ==========================================

    # --- REQUIRED: set these correctly for your data file ---
    # Example:
    # MOF_TYPES  = "1 2 3 4"
    # GAS1_TYPES = "5 6"   # e.g. CO2: C,O
    # GAS2_TYPES = "7 8"   # e.g. CH4: C,H
    # GAS_TYPES  = f"{GAS1_TYPES} {GAS2_TYPES}"

    # ------------------------------
    # Group definitions
    # ------------------------------
    lmp_block(f"""
    group mof     type {MOF_TYPES}
    group gas1    type {GAS1_TYPES}
    group gas2    type {GAS2_TYPES}
    group gas     union gas1 gas2
    group system  union gas mof
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

    # Report gas temperature (important if MOF is frozen)
    lmp_block("""
    compute Tgas gas temp
    # compute_modify Tgas dynamic yes
    thermo_modify temp Tgas
    """)

    # ------------------------------
    # Timestep
    # ------------------------------
    lmp.command(f"timestep {TIMESTEP}")

    # Initial velocities for the whole gas mixture
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

    # Optional geometry optimization
    # lmp.command("fix boxrelax all box/relax iso 0.0 vmax 0.001")
    #  lmp.command("min_style cg")
    #  lmp.command("minimize 1.0e-8 1.0e-4 1000 10000")
    # lmp.command("unfix boxrelax")

    # Run equilibration
    lmp.command(f"run {EQUIL_STEPS}")

    # Clean up equilibration fixes
    #  if sim_type == "rigid":
    #      lmp_block("""
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

    # reset time + new log
    lmp.command("reset_timestep 0")
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
    #      # Re-apply rigid + thermostat
    #      lmp_block(f"""
    #      # Freeze MOF atoms
    #      fix freeze mof setforce 0.0 0.0 0.0
    #      velocity mof set 0.0 0.0 0.0
    #
    #      # Rigid gases (integrator)
    #      fix rig gas rigid/small molecule
    #
    #      # Thermostat rigid bodies
    #      fix tgas gas langevin {TEMP} {TEMP} 0.1 12345
    #      """)
    #  elif sim_type == "flex":
    #      lmp_block(f"""
    #      fix npt_prod system npt temp {TEMP} {TEMP} 0.1 iso {PRES} {PRES} 0.5
    #      """)

    # ------------------------------
    # MSD computation (binary gas, generic)
    # ------------------------------
    lmp_block(f"""
    # ===== MSD for gas1 =====
    compute ch_gas1  gas1 chunk/atom molecule ids once
    compute msdgas1  gas1 msd/chunk ch_gas1
    fix outgas1 all ave/time 100 1 100 c_msdgas1[*] file msd_gas1.data mode vector

    # ===== MSD for gas2 =====
    compute ch_gas2  gas2 chunk/atom molecule ids once
    compute msdgas2  gas2 msd/chunk ch_gas2
    fix outgas2 all ave/time 100 1 100 c_msdgas2[*] file msd_gas2.data mode vector
    """)

    # Run production
    lmp.command(f"run {EMD_STEPS}")

    # Clean up production fixes
    #  if sim_type == "rigid":
    #      lmp_block("""
    #      unfix outgas1
    #      unfix outgas2
    #      unfix tgas
    #      unfix rig
    #      unfix freeze
    #      """)
    #  elif sim_type == "flex":
    #      lmp_block("""
    #      unfix outgas1
    #      unfix outgas2
    #      unfix npt_prod
    #      """)

    print(">>> Run complete.")
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

file_base = args.file_base
model_path = args.model_path
sim_type = args.sim_type.lower()


# Simulation parameters
TEMP = args.temp       # Kelvin
PRES = 1.0
TIMESTEP = 0.0005         # ps (units metal)

EQUIL_STEPS = 20000   # NVT equilibration
if sim_type == "rigid":
    EQUIL_STEPS = 10000   # NVT equilibration
EMD_STEPS = 4000000    # production MD

THERMO_EVERY = 50
DUMP_EVERY = 20


atom_type_pairs_frame = {1: ["Mg", 24.3050], 2: ["O", 15.9994],  3: ["C", 12.0107], 4: ["H", 1.00794]}
atom_type_pairs_gas1 = {5: ["C", 12.0107], 6: ["O", 15.9994]}
atom_type_pairs_gas2 = {7: ["C", 12.0107], 8: ["H", 1.00794]}

specorder = [value[0] for value in list(atom_type_pairs_frame.values())]\
                + [value[0] for value in list(atom_type_pairs_gas1.values())
                                              + list(atom_type_pairs_gas2.values())]

# Atom type mapping (adapt to your data file!)
MOF_TYPES = " ".join(list(str(i) for i in atom_type_pairs_frame.keys()))      # framework atom types 
GAS1_TYPES = " ".join(list(str(i) for i in atom_type_pairs_gas1.keys()))    # C and O atoms of CO2 (example)
GAS2_TYPES = " ".join(list(str(i) for i in atom_type_pairs_gas2.keys()))       # CH4 

run_lammps_nemd()


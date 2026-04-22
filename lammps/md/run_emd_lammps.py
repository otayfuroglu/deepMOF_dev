#
from lammps import lammps
import sys
import argparse
import os


def run_lammps_md():
    lmp = lammps()

    def lmp_block(cmd_block: str):
        for line in cmd_block.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                lmp.command(line)

    # ------------------------------
    # LAMMPS setup
    # ------------------------------
    lmp.command("clear")
    lmp.command("units metal")
    lmp.command("dimension 3")
    lmp.command("boundary p p p")
    lmp.command("atom_style full")
    lmp.command("newton off")

    # ------------------------------
    # Start from data or restart
    # ------------------------------
    if restart_file is not None:
        print(f">>> Restarting from: {restart_file}")
        lmp.command(f"read_restart {restart_file}")
    else:
        print(f">>> Starting from data.{file_base}")
        lmp.command(f"read_data data.{file_base}")
        lmp.command("replicate 1 1 1")

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
    group system union gas mof
    """)

    # ------------------------------
    # Force field
    # ------------------------------
    lmp_block(f"""
    neighbor        2.0 bin
    neigh_modify    every 1 delay 0 check yes

    pair_style      nequip
    pair_coeff * * {model_path} {' '.join(specorder)}
    """)

    # ------------------------------
    # Thermo
    # ------------------------------

    lmp_block(f"""
    thermo          {THERMO_EVERY}
    thermo_style    custom step temp press pe ke etotal vol
    """)

    lmp_block("""
    compute Tgas gas temp
    thermo_modify temp Tgas
    """)

    lmp.command(f"timestep {TIMESTEP}")

    # ------------------------------
    # Fresh start only: create velocities
    # ------------------------------
    if restart_file is None:
        lmp.command(f"velocity gas create {TEMP} 54654 mom yes rot yes dist gaussian")

    # ============================================================
    # EQUILIBRATION (fresh start only)
    # ============================================================
    if restart_file is None:
        print(">>> Equilibration ...")

        if sim_type == "rigid":
            lmp_block(f"""
            fix freeze mof setforce 0.0 0.0 0.0
            velocity mof set 0.0 0.0 0.0
            fix riggas gas rigid/nvt/small molecule temp {TEMP} {TEMP} {100*TIMESTEP}
            """)
        elif sim_type == "flex":
            lmp_block(f"""
            compute mdtemp gas temp
            compute_modify mdtemp dynamic/dof yes
            fix riggas gas rigid/small molecule
            fix_modify riggas dynamic/dof yes
            fix flexmof mof npt temp {TEMP} {TEMP} 100 iso 1.0 1.0 {1000*TIMESTEP}
            """)

        lmp.command(f"log {file_base}_{TEMP}K_restart_eq.log")
        lmp.command(f"run {EQUIL_STEPS}")

        print(">>> Equilibration complete.")

        # keep fixes active into production if you want continuity
        # no unfix here

        print(">>> Production ...")
        lmp.command(f"log {file_base}_{TEMP}K_prod.log")
    else:
        print(">>> Continuing production from restart ...")

        # re-define time integration fixes after read_restart
        if sim_type == "rigid":
            lmp_block(f"""
            fix freeze mof setforce 0.0 0.0 0.0
            velocity mof set 0.0 0.0 0.0
            fix riggas gas rigid/nvt/small molecule temp {TEMP} {TEMP} {100*TIMESTEP}
            """)
        elif sim_type == "flex":
            lmp_block(f"""
            compute mdtemp gas temp
            compute_modify mdtemp dynamic/dof yes
            fix riggas gas rigid/small molecule
            fix_modify riggas dynamic/dof yes
            fix flexmof mof npt temp {TEMP} {TEMP} 100 iso 1.0 1.0 {1000*TIMESTEP}
            """)

        lmp.command(f"log {file_base}_{TEMP}K_prod.log append")

    # ------------------------------
    # Restart writing
    # ------------------------------
    # two alternating restart files is safer than one file
    lmp.command(
        f"restart 500000 {file_base}_{TEMP}K.restart1 {file_base}_{TEMP}K.restart2"
    )

    # ------------------------------
    # Dump
    # ------------------------------

    dump_file = f"{file_base}_{TEMP}K.lammpstrj"

    lmp_block(f"""
    dump dump_all all custom {DUMP_EVERY} {dump_file} id mol type xu yu zu
    dump_modify dump_all sort id append yes
    """)

    # ------------------------------
    # MSD computation
    # ------------------------------
    if restart_file is None:
        msd_file = "msd_gas.data"
    else:
        restart_tag = os.path.basename(restart_file)
        msd_file = f"msd_gas_{restart_tag}.data"

    lmp_block(f"""
    compute ch_gas gas chunk/atom molecule ids once
    compute msdgas gas msd/chunk ch_gas
    fix outgas gas ave/time 100 1 100 c_msdgas[*] file {msd_file} mode vector
    """)

    # ------------------------------
    # Production / continuation run
    # ------------------------------
    run_steps = CONTINUE_STEPS if restart_file is not None else PROD_STEPS
    lmp.command(f"run {run_steps}")

    print(">>> Run complete.")
    lmp.close()


parser = argparse.ArgumentParser(description="Run or restart LAMMPS MD")
parser.add_argument("-file_base", type=str, required=True)
parser.add_argument("-model_path", type=str, required=True)
parser.add_argument("-sim_type", type=str, required=True)
parser.add_argument("-gas_type", type=str, required=True)
parser.add_argument("-temp", type=int, required=True)
parser.add_argument("-restart_file", type=str, default=None,
                    help="Restart file to continue from")
parser.add_argument("-continue_steps", type=int, default=500000,
                    help="Steps to run after restart")
args = parser.parse_args()

file_base = args.file_base
model_path = args.model_path
sim_type = args.sim_type.lower()
gas_type = args.gas_type.lower()
restart_file = args.restart_file
CONTINUE_STEPS = args.continue_steps

TEMP = args.temp
PRES = 1.0
TIMESTEP = 0.001

EQUIL_STEPS = 20000
if sim_type == "rigid":
    EQUIL_STEPS = 10000
PROD_STEPS = 2000000

THERMO_EVERY = 50
DUMP_EVERY = 20

atom_type_pairs_frame = {
    1: ["Mg", 24.3050],
    2: ["O", 15.9994],
    3: ["C", 12.0107],
    4: ["H", 1.00794],
}

if gas_type == "co2":
    atom_type_pairs_gas = {5: ["C", 12.0107], 6: ["O", 15.9994]}
elif gas_type == "ch4":
    atom_type_pairs_gas = {5: ["C", 12.0107], 6: ["H", 1.00794]}
else:
    print("Please enter gas type as 'CH4' or 'CO2'")
    sys.exit(1)

specorder = [value[0] for value in atom_type_pairs_frame.values()] + \
            [value[0] for value in atom_type_pairs_gas.values()]

MOF_TYPES = " ".join(str(i) for i in atom_type_pairs_frame.keys())
GAS_TYPES = " ".join(str(i) for i in atom_type_pairs_gas.keys())

run_lammps_md()

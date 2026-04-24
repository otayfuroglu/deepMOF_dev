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
        for key, values in atom_type_pairs_gas1.items():
            lmp.command(f"mass {key} {values[1]}")
        for key, values in atom_type_pairs_gas2.items():
            lmp.command(f"mass {key} {values[1]}")

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
        lmp.command(f"log {file_base}_{TEMP}K_eq.log")

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

        lmp.command(f"run {EQUIL_STEPS}")
        print(">>> Equilibration complete.")

    else:
        print(">>> Continuing production from restart ...")

        # Re-define fixes after read_restart
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

    # ============================================================
    # PRODUCTION / CONTINUATION
    # ============================================================
    print(">>> Production ...")

    # keep timestep continuity on restart
    if restart_file is None:
        lmp.command("reset_timestep 0")

    if restart_file is None:
        lmp.command(f"log {file_base}_{TEMP}K_prod.log")
    else:
        lmp.command(f"log {file_base}_{TEMP}K_prod.log append")

    # ------------------------------
    # Restart writing
    # ------------------------------
    lmp.command(
        f"restart 100000 {file_base}_{TEMP}K.restart"
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
        msd_gas1_file = "msd_gas1.data"
        msd_gas2_file = "msd_gas2.data"
    else:
        restart_tag = os.path.basename(restart_file)
        msd_gas1_file = f"msd_gas1_{restart_tag}.data"
        msd_gas2_file = f"msd_gas2_{restart_tag}.data"

    lmp_block(f"""
    compute ch_gas1  gas1 chunk/atom molecule ids once
    compute msdgas1  gas1 msd/chunk ch_gas1
    fix outgas1 gas1 ave/time 100 1 100 c_msdgas1[*] file {msd_gas1_file} mode vector

    compute ch_gas2  gas2 chunk/atom molecule ids once
    compute msdgas2  gas2 msd/chunk ch_gas2
    fix outgas2 gas2 ave/time 100 1 100 c_msdgas2[*] file {msd_gas2_file} mode vector
    """)

    # ------------------------------
    # Run
    # ------------------------------
    run_steps = CONTINUE_STEPS if restart_file is not None else PROD_STEPS
    lmp.command(f"run {run_steps}")

    print(">>> Run complete.")
    lmp.close()


parser = argparse.ArgumentParser(description="Run or restart binary LAMMPS MD")
parser.add_argument("-file_base", type=str, required=True)
parser.add_argument("-model_path", type=str, required=True)
parser.add_argument("-sim_type", type=str, required=True)
parser.add_argument("-temp", type=int, required=True)
parser.add_argument("-restart_file", type=str, default=None,
                    help="Restart file to continue from")
parser.add_argument("-continue_steps", type=int, default=500000,
                    help="Steps to run after restart")
args = parser.parse_args()

file_base = args.file_base
model_path = args.model_path
sim_type = args.sim_type.lower()
restart_file = args.restart_file
CONTINUE_STEPS = args.continue_steps

TEMP = args.temp
PRES = 1.0
TIMESTEP = 0.001

EQUIL_STEPS = 20000
if sim_type == "rigid":
    EQUIL_STEPS = 10000
PROD_STEPS = 4000000

THERMO_EVERY = 50
DUMP_EVERY = 20

atom_type_pairs_frame = {
    1: ["Mg", 24.3050],
    2: ["O", 15.9994],
    3: ["C", 12.0107],
    4: ["H", 1.00794],
}
atom_type_pairs_gas1 = {
    5: ["C", 12.0107],
    6: ["O", 15.9994],
}
atom_type_pairs_gas2 = {
    7: ["C", 12.0107],
    8: ["H", 1.00794],
}

specorder = (
    [value[0] for value in atom_type_pairs_frame.values()]
    + [value[0] for value in atom_type_pairs_gas1.values()]
    + [value[0] for value in atom_type_pairs_gas2.values()]
)

MOF_TYPES = " ".join(str(i) for i in atom_type_pairs_frame.keys())
GAS1_TYPES = " ".join(str(i) for i in atom_type_pairs_gas1.keys())
GAS2_TYPES = " ".join(str(i) for i in atom_type_pairs_gas2.keys())

run_lammps_md()

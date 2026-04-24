from lammps import lammps
import sys
import argparse
import os

import re
import shutil
from pathlib import Path



def get_restart_step_from_filename(restart_file):
    """
    Extract restart timestep from filename.

    Example:
      MgMOF74_clean_fromCORE_36CO2_2CH4_298K.1000 -> 1000
    """
    name = Path(restart_file).name
    m = re.search(r"\.(\d+)$", name)
    if not m:
        raise ValueError(
            f"Could not extract restart timestep from filename: {restart_file}\n"
            "Expected format like: system_name.1000"
        )
    return int(m.group(1))


def trim_lammpstrj_in_place_before_restart(traj_file, restart_file):
    """
    Keep only trajectory frames with timestep <= restart timestep.
    Creates backup before overwriting.
    """
    traj = Path(traj_file)
    if not traj.exists():
        print(f">>> Trajectory not found, skipping trim: {traj}")
        return

    max_step = get_restart_step_from_filename(restart_file)

    backup = traj.with_suffix(traj.suffix + ".bak")
    tmp = traj.with_suffix(traj.suffix + ".tmp")

    shutil.copy2(traj, backup)

    kept = 0
    removed = 0

    with open(traj, "r") as fin, open(tmp, "w") as fout:
        while True:
            line = fin.readline()
            if not line:
                break

            if not line.startswith("ITEM: TIMESTEP"):
                continue

            frame = [line]

            step_line = fin.readline()
            if not step_line:
                break
            frame.append(step_line)
            step = int(step_line.strip())

            # ITEM: NUMBER OF ATOMS + natoms
            line = fin.readline()
            frame.append(line)
            nat_line = fin.readline()
            frame.append(nat_line)
            natoms = int(nat_line.strip())

            # ITEM: BOX BOUNDS + 3 box lines
            for _ in range(4):
                frame.append(fin.readline())

            # ITEM: ATOMS + atom lines
            frame.append(fin.readline())
            for _ in range(natoms):
                frame.append(fin.readline())

            if step <= max_step:
                fout.writelines(frame)
                kept += 1
            else:
                removed += 1

    tmp.replace(traj)

    print(f">>> Restart step from filename : {max_step}")
    print(f">>> Trajectory backup written  : {backup}")
    print(f">>> Trimmed trajectory         : {traj}")
    print(f">>> Frames kept/removed        : {kept}/{removed}")


def trim_log_in_place_before_restart(log_file, restart_file):
    """
    Trim LAMMPS log file so it only contains data up to restart timestep.
    Creates a backup before overwriting.
    """
    log = Path(log_file)
    if not log.exists():
        print(f">>> Log file not found, skipping trim: {log}")
        return

    max_step = get_restart_step_from_filename(restart_file)

    backup = log.with_suffix(log.suffix + ".bak")
    tmp = log.with_suffix(log.suffix + ".tmp")

    shutil.copy2(log, backup)

    kept_lines = []
    last_valid_index = None

    with open(log, "r") as f:
        lines = f.readlines()

    # find last occurrence of timestep <= restart step
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if not parts:
            continue

        # thermo lines usually start with timestep
        if parts[0].isdigit():
            step = int(parts[0])
            if step <= max_step:
                last_valid_index = i

    if last_valid_index is None:
        print(">>> WARNING: No valid timestep found in log; keeping full log.")
        return

    # keep everything up to that line
    kept_lines = lines[: last_valid_index + 1]

    with open(tmp, "w") as f:
        f.writelines(kept_lines)

    tmp.replace(log)

    print(f">>> Log backup written        : {backup}")
    print(f">>> Trimmed log               : {log}")
    print(f">>> Last timestep kept        : {max_step}")


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

    log_file_prod = f"{file_base}_{TEMP}K_prod.log"
    if restart_file is None:
        lmp.command(f"log {log_file_prod}")
    else:
        trim_log_in_place_before_restart(log_file_prod, restart_file)
        lmp.command(f"log {log_file_prod} append")

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
    if restart_file is not None:
        trim_lammpstrj_in_place_before_restart(dump_file, restart_file)

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

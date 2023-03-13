#  from ase.io.lammpsrun import read_lammps_dump_text
from ase.io import read
from ase import units
import numpy as np

import tqdm
import argparse
import os
import shutil

from pathlib import Path
from lammpsTraj2n2p2 import aseDb2Runner


n_cpu = 20

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-calc_type", type=str, required=True)
parser.add_argument("-metal", type=str, required=True)
parser.add_argument("-temp", type=int, required=True)
#  parser.add_argument("-geom", type=str, required=True)
parser.add_argument("-idx", type=int, required=True)
parser.add_argument("-tmpdir", type=str, required=False)
#  parser.add_argument("-memory", type=int, required=True)
args = parser.parse_args()

#  init_geom = "optimized"
#  calc_type = "md"
#  metal = "Li"
#  temp = 200
metal_mass_pair = {"Li": 6.94000000}
calc_type = args.calc_type
metal = args.metal
temp = args.temp
#  init_geom = args.geom
#  memory = args.memory

BASE_DIR = "/kernph/tayfur0000/works/alanates"
GEOMS_DIR = f"{BASE_DIR}/Top20"
MD_DIR = f"{BASE_DIR}/n2p2/runMD/lammps_md_{temp}K"
WORKS_DIR = f"{BASE_DIR}/n2p2/runSelect/{calc_type}_{temp}K"

MODEL_DIR_1 = f"{BASE_DIR}/n2p2/runTrain/zeta_12416_frate10_24_task_14kdata/best_epoch_90"
MODEL_DIR_2 = f"{BASE_DIR}/n2p2/runTrain/zeta_12416_frate10_24_task_14kdata_nnp2/best_epoch_60"

MD_DIRS = []
OUT_DIRS = []
geoms_path = []
for _DIR in [item for item in os.listdir(GEOMS_DIR) if metal in item]:
    for struct_type in ["polymeric", "isolated"]:
    #  for struct_type in ["isolated"]:
        GEOM_DIR = f"{GEOMS_DIR}/{_DIR}/{struct_type}"
        for fl_name in os.listdir(GEOM_DIR):
            fl_base = fl_name.replace('.ascii', '')
            MD_DIRS += [f"{MD_DIR}/{_DIR}/{struct_type}/{fl_base}"]
            OUT_DIRS += [f"{WORKS_DIR}/{_DIR}/{struct_type}/{fl_base}"]


atom_type_number_pair = {1:13, 2:3, 3:1} # Al, Li, H

idx = args.idx

if idx > (len(OUT_DIRS) - 1):
        quit()

OUT_DIR = OUT_DIRS[idx]
MD_DIR = MD_DIRS[idx]
#  TMP_DIR = args.tmpdir
#  geom_path = geoms_path[idx]

if not os.path.exists(OUT_DIR):
#  if  os.path.exists(OUT_DIR):
        #  CURRENT_DIR = os.getcwd()

        # create outpur folders
        # easy create nested forder wether it is exists
        Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        # change to local scratch directory
        #  os.chdir(TMP_DIR)
        os.chdir(OUT_DIR)

        traj_name = [fl for fl in os.listdir(MD_DIR) if ".lammpstrj" in fl][0]
        traj = read(f"{MD_DIR}/{traj_name}", format="lammps-dump-text", index=":")
        atomic_numbers = [atom_type_number_pair[key] for key in traj[0].get_atomic_numbers()]

        aseDb2Runner(traj, atomic_numbers)

        for i, MODEL_DIR in enumerate([MODEL_DIR_1, MODEL_DIR_2]):
            Path(f"{OUT_DIR}/nnp-data-{i+1}").mkdir(parents=True, exist_ok=True)
            for fl in os.listdir(MODEL_DIR):
                shutil.copy(f"{MODEL_DIR}/{fl}", f"{OUT_DIR}/nnp-data-{i+1}")

        os.system(f"mpirun -np {n_cpu} /kernph/tayfur0000/n2p2/bin/nnp-comp2 compare > log_compare.out")
        os.system(f"/kernph/tayfur0000/n2p2/bin/nnp-comp2 select 4 0.001 0.01 Al Li H > log_select.out")
        os.system(f"cat comp-selection.data >> {WORKS_DIR}/selected.data")
        os.system(f"cat comp-selection.data >> {WORKS_DIR}/../all_selected.data")


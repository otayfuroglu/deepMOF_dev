
import os, sys
sys.path.insert(0, "/truba_scratch/otayfuroglu/deepMOF_dev")
from calculateGeomWithQM import CaculateData
import multiprocessing
#  import getpass
import argparse
from pathlib import Path
from ase.io import read


#  USER = getpass.getuser()

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-in_extxyz", type=str, required=True)
parser.add_argument("-orca_path", type=str, required=True)
parser.add_argument("-calc_type", type=str, required=True)
parser.add_argument("-calculator_type", type=str, required=True)
parser.add_argument("-n_core", type=int, required=True)
args = parser.parse_args()


cwd = os.getcwd()

n_core = args.n_core
orca_path = args.orca_path
calc_type = args.calc_type
calculator_type = args.calculator_type

in_extxyz = args.in_extxyz
in_extxyz = in_extxyz.split('/')[-1]
in_extxyz_path = f"{cwd}/{in_extxyz}"
#  out_extxyz = "/".join(in_extxyz[0:-1]) + "/sp_" + in_extxyz[-1]
out_extxyz_path = f"{cwd}/{calc_type}_{in_extxyz}"
#  csv_path = in_extxyz.replace(".extxyz", ".csv")
csv_path = f"{cwd}/{calc_type}_{in_extxyz.replace('.extxyz', '.csv')}"
#  OUT_DIR = "run_" + in_extxyz[-1].split(".")[0]
#  os.chdir(os.getcwd())

# set default
n_task = 8

atoms_list = read(in_extxyz_path, index=":")
if calculator_type == "g16":
    n_task = 1
elif len(atoms_list) == 1:
    n_task = n_core
else:
    if n_core == 20:
        n_task = 10
    if n_core == 24 or n_core == 48:
        n_task = 6
    if n_core == 40 or n_core == 80:
        n_task = 8
    if n_core == 28 or n_core == 56:
        n_task = 4
    if n_core == 112:
        #  n_task = 16
        n_task = 16
    if n_core == 110:
        #  n_task = 10
        n_task = 22
        #  n_task = 54

n_proc = int(n_core / n_task)

properties = ["energy", "forces", "dipole_moment"]

#  Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
#  os.chdir(OUT_DIR)

calculate = CaculateData(orca_path, calc_type, calculator_type,
                         n_task, in_extxyz_path, out_extxyz_path,
                         csv_path, rm_out_dir=True)
print ("Nuber of out of range geomtries", calculate.countAtoms())
print("QM calculations Running...")
# set remove file if has error
#  calculate.rmNotConvFiles()
calculate.calculate_data(n_proc)
print("DONE")
#  print("All process taken %2f minutes" %((time.time()- start) / 60.0))

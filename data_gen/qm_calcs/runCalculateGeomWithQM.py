
import os, sys
sys.path.insert(0, "/truba_scratch/otayfuroglu/deepMOF_dev")
from calculateGeomWithQM import CaculateData
import multiprocessing
#  import getpass
import argparse
from pathlib import Path


#  USER = getpass.getuser()

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-in_extxyz", type=str, required=True)
parser.add_argument("-n_core", type=int, required=True)
args = parser.parse_args()


in_extxyz = args.in_extxyz
in_extxyz_split = in_extxyz.split('/')
out_extxyz = "/".join(in_extxyz_split[0:-1]) + "/sp_" + in_extxyz_split[-1]
csv_path = in_extxyz.replace(".extxyz", ".csv")
OUT_DIR = "run_" + in_extxyz_split[-1].split(".")[0]
os.chdir(os.getcwd())

n_core = args.n_core

# set default
n_task = 8

if n_core == 24 or n_core == 48:
    n_task = 6
if n_core == 40 or n_core == 80:
    n_task = 8
if n_core == 28 or n_core == 56:
    n_task = 4
if n_core == 112:
    n_task = 8

n_proc = int(n_core / n_task)

properties = ["energy", "forces", "dipole_moment"]

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
os.chdir(OUT_DIR)

calculate = CaculateData(properties, n_task, in_extxyz, out_extxyz, csv_path)
print ("Nuber of out of range geomtries", calculate.countAtoms())
print("QM calculations Running...")
# set remove file if has error
#  calculate.rmNotConvFiles()
calculate.calculate_data(n_proc)
print("DONE")
#  print("All process taken %2f minutes" %((time.time()- start) / 60.0))

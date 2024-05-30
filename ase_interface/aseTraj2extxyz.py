#
from ase import Atoms
from ase.io import write
from ase.io.trajectory import Trajectory

from tqdm import tqdm

import argparse
import os


parser = argparse.ArgumentParser(description="Give samefile.traj file name")
parser.add_argument("-trajFile", "--trajFile", type=str, required=True, help="give traj file with full path")
parser.add_argument("-interval", "--interval", type=int, required=True, help="give interval")

args = parser.parse_args()
trajFile_path = args.trajFile
interval = args.interval
ase_trj = Trajectory(trajFile_path)

outFile_path = trajFile_path.split("/")[-1].split(".")[0] + ".extxyz"
if os.path.exists(outFile_path):
    os.remove(outFile_path)

for i, atoms in tqdm(enumerate(ase_trj), total=len(ase_trj)):
    if i % interval == 0:
        atoms.info["label"] =  "frame_" + "{0:0>5}".format(i)
        write(outFile_path, atoms, append=True)

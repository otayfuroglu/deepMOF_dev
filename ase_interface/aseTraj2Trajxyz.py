#
import os, io
import numpy as np
import shutil
import argparse

from ase import Atoms
from ase.io import write
from ase.io.trajectory import Trajectory

from tqdm import tqdm

def get_trajectory(data, outFile_path, interval):
    #get current working directory and make a scratch 
    #directory
    path = './.scratch'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    #write each structure from the .traj file in .xyz format
    for i, atoms in tqdm(enumerate(data), total=len(data)):
        if i % interval == 0:
            string = 'structure%03d' % (i,) +'.xyz'
            outStruct = os.path.join(path, string)
            write(outStruct, atoms)
            #combines all optimization structures in one trajectory 
            #file
            inFile = open(os.path.join(path, 'structure%03d' %
                          (i,)  +'.xyz'), 'r')
            outFile = open(outFile_path, 'a')
            for j, line in enumerate(inFile.readlines()):
                if j == 1:
                    line = "Created by ASE\n"
                outFile.write(line)
    shutil.rmtree(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Give samefile.traj file name")
    parser.add_argument("-interval", "--interval", type=int, required=True, help="give interval")
    parser.add_argument("-trajFile", "--trajFile", type=str, required=True, help="give traj file with full path")

    args = parser.parse_args()
    trajFile_path = args.trajFile
    data = Trajectory(trajFile_path)

    outFile_path = trajFile_path[:-5] + ".xyz"
    if os.path.exists(outFile_path):
        os.remove(outFile_path)
    get_trajectory(data, outFile_path, args.interval)

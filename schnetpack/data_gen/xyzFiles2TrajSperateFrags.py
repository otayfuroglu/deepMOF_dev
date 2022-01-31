#! /truba/home/yzorlu/miniconda3/bin/python

from ase import Atoms
from ase.io import write
import os, io
import numpy as np
import shutil
import argparse



def main(xyzDIR, fragBase):
    #output file name

    fileNames = sorted([item for item in os.listdir(xyzDIR) if item.endswith(".xyz")])
    print("Nuber of xyx files --> ", len(fileNames))

    for i in range(1, 11):
        fragName = fragBase+str(i)
        print(fragName)
        outFileName = fragName+"_traj.xyz"
        for i, fileName in enumerate(fileNames):
            if fileName.startswith(fragName):

                if i == 0:
                    assert not os.path.exists(fragName+outFileName), "%s exists" %(fragName+outFileName)

                inFile = open(os.path.join(xyzDIR, fileName), "r")
                fileStr = inFile.read()
                outFile = open(outFileName, 'a')
                outFile.write(fileStr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-xyzDIR", "--xyzDIR", type=str, required=True, help="give xyz files directory")
    parser.add_argument("-fragBASE", "--fragBASE", type=str, required=True, help="give fragmemt base name")

    args = parser.parse_args()
    xyzDIR = args.xyzDIR
    fragBASE = args.fragBASE
    main(xyzDIR, fragBASE)

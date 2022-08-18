#! /truba/home/yzorlu/miniconda3/bin/python -u

import pandas as pd
import numpy as np
import lammps_logfile
import argparse, os


def calcLengths(volumes):
    return np.cbrt(volumes.mean())


def calcMeanLnVolumes(volumes):
    return np.log(volumes).mean()


def calcBulkMod(df):
    from ase.units import kJ

    volumes = df["Volume"].to_numpy()
    mean_volumes = volumes.mean()
    sqr_mean_volumes = (volumes ** 2).mean()

    kB = 8.617333e-5 # in eV/K

    B0 = kB * args.temp * (mean_volumes / (sqr_mean_volumes - mean_volumes ** 2)) # in eV/A^3
    B0 = B0 / kJ * 1.0e24 #  in 'GPa'
    print("Mean a:", calcLengths(volumes))
    print("Bulk Modulus: ", B0)


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-log", type=str, required=True)
parser.add_argument("-skip", type=int, required=True)
parser.add_argument("-temp", type=float, required=True)
args = parser.parse_args()


log_path = args.log
log_base = os.path.basename(log_path).split(".")[0]
log = lammps_logfile.File(log_path)

data = pd.DataFrame()
for label in log.get_keywords():
    data[label] = log.get(label)
#  df.to_csv("test.csv")

n_frame_atoms = data["Atoms"][0]
initial_skip = args.skip
#  print(len(data))
data = data[initial_skip:]
calcBulkMod(df=data)



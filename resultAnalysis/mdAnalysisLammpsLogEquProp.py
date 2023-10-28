#! /truba/home/yzorlu/miniconda3/bin/python -u

import lammps_logfile
import pandas as pd
import numpy as np

import os
import argparse

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-log", type=str, required=True)
#  parser.add_argument("-stepsize", type=float, required=True)
parser.add_argument("-skip", type=int, required=True)
args = parser.parse_args()


def calcLengths(volumes):
    return np.cbrt(volumes.mean())


def calcBulkMod(df):
    from ase.units import kJ

    volumes = df["Volume"].to_numpy()
    mean_volumes = volumes.mean()
    sqr_mean_volumes = (volumes ** 2).mean()

    kB = 8.617333e-5 # in eV/K

    B0 = kB * temp * (mean_volumes / (sqr_mean_volumes - mean_volumes ** 2)) # in eV/A^3
    B0 = B0 / kJ * 1.0e24 #  in 'GPa'
    print("Mean a:", calcLengths(volumes))
    print("Bulk Modulus: ", B0)


log_path = args.log
log_base = os.path.basename(log_path).split(".")[0]
log = lammps_logfile.File(log_path)

#  step_size = args.stepsize # fs
initial_skip = args.skip

df = pd.DataFrame()
for label in log.get_keywords():
    df[label] = log.get(label)

df = df[initial_skip:]
temp = 100
calcBulkMod(df)

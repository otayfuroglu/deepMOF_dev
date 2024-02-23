from ase.calculators.vasp import Vasp
from ase.io.extxyz import read_extxyz
from ase.io import read, write
from ase import Atoms
from ase.io.vasp import read_vasp_out

from pathlib import Path
import os
import argparse
import pandas as pd


import numpy as np
from ase.thermochemistry import HarmonicThermo
from ase.vibrations import Vibrations
from ase import units



parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-extxyz_path", type=str, required=True, help="..")
args = parser.parse_args()
extxyz_path = args.extxyz_path


if "isolated" in extxyz_path:
    keyword = "isolated"
elif "polymeric" in extxyz_path:
    keyword = "polymeric"
else:
    keyword = "test"


h = 0.01

keyword += "_24atoms_4x3x4"

atoms_list = read(extxyz_path, index=":")

df_helmholtz = pd.DataFrame()
temperature_list = range(0, 1001, 10)
df_helmholtz["Temperature"] = temperature_list
for i, atoms in enumerate(atoms_list):
    name = atoms.info["label"]
    potentialenergy = atoms.get_potential_energy()
    BASE_DIR = Path(f"{keyword}")
    print(name)

    #  try:
    vib = Vibrations(atoms, name=f"{BASE_DIR}/{name}/vib/")
    vib_energies = vib.get_energies()
        #  print(vib_energies)
    #  except:
        #  continue

    #  quit()

    vib_energies = [mode for mode in np.real(vib_energies) if mode > 1e-3]
    #  vib_energies = [complex(1.0e-8, 0) if energy < 1.0e-4 else energy for energy in vib_energies]

    #  print(vib_energies)
    # get free energy from ase
    #  free_energy_class = HarmonicThermo(vib_energies, potentialenergy=0.)#potentialenergy)
    free_energy_class = HarmonicThermo(vib_energies, potentialenergy=potentialenergy)
    #  df_gibbs["Temperature"] = temperature_list
    helm_holtz_energies = []
    for temperature in temperature_list:
        print(temperature)
        if temperature == 0:
            #  temperature = 1e-5
            temperature = 1
        helm_holtz_energy = free_energy_class.get_helmholtz_energy(temperature, verbose=True)
        helm_holtz_energies += [helm_holtz_energy / len(atoms)]
    df_helmholtz[f"structure_{i}"] = helm_holtz_energies
    #  df_gibbs[f"structure_{i}"] = gibbs

df_helmholtz.to_csv(f"{BASE_DIR}/helmholtz_{keyword}.csv")
#  df_gibbs.to_csv(f"gibbs_{keyword}.csv")


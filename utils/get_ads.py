from ase import Atoms
from ase.io import read, write
from itertools import combinations
import numpy as np


def extract_ads(system, ads_atoms, n_frame):
    ref_symbols =ads_atoms.get_chemical_symbols()
    extracted_ads_atoms_list = []
    ads_indices = np.array(range(len(ads_atoms))) + n_frame
    i = ads_indices[-1]

    while ads_indices[-1] < len(system) :
        symbols = [system[i].symbol for i in ads_indices]
        if symbols == ref_symbols:
            extracted_ads_atoms = system[list(ads_indices)]
            extracted_ads_atoms_list.append(extracted_ads_atoms)
        ads_indices += 1
    write("extracted_ch4.extxyz", extracted_ads_atoms_list)


def extract_ads_indices(system, ads_atoms, n_frame):
    ref_symbols =ads_atoms.get_chemical_symbols()
    ads_indices = np.array(range(len(ads_atoms))) + n_frame
    i = ads_indices[-1]

    extracted_ads_indices = []
    while ads_indices[-1] < len(system) :
        symbols = [system[i].symbol for i in ads_indices]
        if symbols == ref_symbols:
            extracted_ads_indices += [ads_indices.tolist()]
        ads_indices += 1
    return  extracted_ads_indices

# Load system and reference CO2
n_frame = 81
system = read('MgF2_CO2_CH4_0.extxyz')          # The full atomic system
ads_atoms = read('CO2.xyz') # CO2 molecule
#  ads_atoms = read('CH4.xyz') # CO2 molecule

ads_indices_list = extract_ads_indices(system, ads_atoms, n_frame)

from itertools import combinations
for ads_indices in ads_indices_list:
    fix_bond_indices = list(combinations(ads_indices, 2))[:len(ads_indices)-1]
    print(fix_bond_indices)


import sys, os
import numpy as np
from yaff import *
log.set_level(0)
from ase import Atoms
from ase.data import vdw_radii
from ase import *
from ase.io import *
from glob import *
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import argparse

from insertionAds import insertMixAds



def scale_vdw(atom_nums, sf_vdw):
    for num in atom_nums:
        vdw_radii[num] = vdw_radii[num] * sf_vdw
        print(f"vdw adjusted to {vdw_radii[num]} for {num}")


def create_bulk_mix_fluid(atoms, atoms2,  max_n_ads, pbc=False):

    #NOTE scaled when call to avoid rescale
    # scale vdw
    #  scale_vdw(atom_nums, sf_vdw)

    #  atoms.center(vacuum=2.5)
    atoms.center(vacuum=6)
    structure = System(numbers = atoms.get_atomic_numbers(),
                        pos = atoms.get_positions()*angstrom,
                        rvecs = atoms.get_cell()*angstrom)
    structure.detect_bonds()
    loading = insertMixAds(structure, ads, ads2, vdw_radii, pbc=pbc)
    n_written = loading.load_mix_fixed_nads(n_trial=50000, nads=nads1, nads2=nads2)
    print('Written %d adsorbates'%n_written)
    out_path = f"bulk_mixture_{'_'.join([name.split('.')[0] for name in [fluid_path, fluid2_path]])}.extxyz"
    loading.write_output(out_path, append=True)
    loading.print_ratio()
    #  break


def ins_fluid(fl_name, atoms, max_n_ads, pbc=False):

    #NOTE scaled when call to avoid rescale
    #  vdw_radii[1] = 1.0
    #  vdw_radii[6] = 1.25
    #  vdw_radii[7] = 1.50
    #  vdw_radii[8] = 1.50


        atoms.center(vacuum=0.0)


        structure = System(numbers = atoms.get_atomic_numbers(),
                            pos = atoms.get_positions()*angstrom,
                            rvecs = atoms.get_cell()*angstrom)
        structure.detect_bonds()
        loading = insertMixAds(structure, ads, ads2, vdw_radii, pbc=pbc)
        n_written = loading.load_mix_fixed_nads(n_trial=10000, nads=nads1, nads2=nads2)
        # bigger cell for adding ads
        if not n_written:
            atoms.center(vacuum=0.0)
            structure = System(numbers = atoms.get_atomic_numbers(),
                                pos = atoms.get_positions()*angstrom,
                                rvecs = atoms.get_cell()*angstrom)
            structure.detect_bonds()
            loading = insertMixAds(structure, ads, ads2, vdw_radii, pbc=pbc)
            n_written = loading.load_mix_fixed_nads(n_trial=10000, nads=nads1, nads2=nads2)
        print('Written %d adsorbates'%n_written)
        #  out_path = f"{out_dir}/{fl_name}"
        out_path = f"{'_'.join([name.split('.')[0] for name in [fl_name, fluid_path, fluid2_path]])}_{nads1}_{nads2}.extxyz"
        loading.write_output(out_path, append=True)
        loading.print_ratio()


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-struc_dir", type=str, required=True,)
parser.add_argument("-out_dir", type=str, required=True,)
parser.add_argument("-fluid1_path", type=str, required=True,)
parser.add_argument("-fluid2_path", type=str, required=True,)
parser.add_argument("-sf_vdw", type=float, required=True,)
parser.add_argument("-nads1", type=int, required=True,)
parser.add_argument("-nads2", type=int, required=True,)

args = parser.parse_args()
struc_dir = args.struc_dir
out_dir = args.out_dir
fluid_path = args.fluid1_path
fluid2_path = args.fluid2_path
sf_vdw = args.sf_vdw
nads1 = args.nads1
nads2 = args.nads2

vdw_radii = vdw_radii.copy()

ads = System.from_file(fluid_path)
ads.detect_bonds()

ads2 = System.from_file(fluid2_path)
ads2.detect_bonds()


# run create bulk mixture gas
def run_bulk_mixture():
    for i in range(25):
        #  atoms = read(fluid_path) + read(fluid2_path)
        atoms = read(fluid_path)
        atoms2 = read(fluid2_path)
        atom_nums = set((atoms + atoms2).numbers)
        if i == 0:
            # scale vdw
            scale_vdw(atom_nums, sf_vdw)
        create_bulk_mix_fluid(atoms, atoms2, max_n_ads=1000, pbc=False)


def run_ins_mixture():
    for i, fl_name in enumerate([fl for fl in os.listdir(struc_dir) if fl.endswith(".xyz")]):
        atoms = read(f"{struc_dir}/{fl_name}")
        if i == 0:
            # scale vdw
            scale_vdw(set(atoms.numbers), sf_vdw)
        for _ in range(25):
            ins_fluid(fl_name, atoms, max_n_ads=200, pbc=False)
        break

#  run_bulk_mixture()
run_ins_mixture()

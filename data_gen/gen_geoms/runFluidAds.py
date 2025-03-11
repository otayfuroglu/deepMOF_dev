
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

from insertionAds import insertAds



def scale_vdw(atoms, sf_vdw):
    for num in set(atoms.numbers):
        vdw_radii[num] = vdw_radii[num] * sf_vdw
        print(f"vdw adjusted to {vdw_radii[num]} for {num}")


def create_bulk_fluid(max_n_ads, pbc=False):

    atoms = read(fluid_path)

    # scale vdw
    scale_vdw(atoms, sf_vdw)

    atoms.center(vacuum=2.5)
    structure = System(numbers = atoms.get_atomic_numbers(),
                        pos = atoms.get_positions()*angstrom,
                        rvecs = atoms.get_cell()*angstrom)
    structure.detect_bonds()
    loading = insertAds(structure, ads, vdw_radii, pbc=pbc)
    n_written = loading.load(n_trial=100000, n_load=max_n_ads)
    print('Written %d adsorbates'%n_written)
    #  out_path = f"{out_dir}/{fl_name}"
    out_path = f"{out_dir}/bulk_fluid.extxyz"
    loading.write_output(out_path, append=False)
    #  break


def ins_fluid(max_n_ads, pbc=False):
    #  vdw_radii[1] = 1.0
    #  vdw_radii[6] = 1.25
    #  vdw_radii[7] = 1.50
    #  vdw_radii[8] = 1.50


    for fl_name in [fl for fl in os.listdir(struc_dir) if fl.endswith(".extxyz")]:
        atoms = read(f"{struc_dir}/{fl_name}")
        atoms.center(vacuum=5)

        # scale vdw
        scale_vdw(atoms, sf_vdw)

        structure = System(numbers = atoms.get_atomic_numbers(),
                            pos = atoms.get_positions()*angstrom,
                            rvecs = atoms.get_cell()*angstrom)
        structure.detect_bonds()
        loading = insertAds(structure, ads, vdw_radii, pbc=pbc)
        n_written = loading.load(n_trial=10000, n_load=max_n_ads)
        # bigger cell for adding ads
        if not n_written:
            atoms.center(vacuum=1.0)
            structure = System(numbers = atoms.get_atomic_numbers(),
                                pos = atoms.get_positions()*angstrom,
                                rvecs = atoms.get_cell()*angstrom)
            structure.detect_bonds()
            loading = insertAds(structure, ads, vdw_radii, pbc=pbc)
            n_written = loading.load(n_trial=10000, n_load=max_n_ads)
        print('Written %d adsorbates'%n_written)
        #  out_path = f"{out_dir}/{fl_name}"
        out_path = f"{'_'.join([name.split('.')[0] for name in [fl_name, fluid_path]])}.extxyz"
        loading.write_output(out_path, append=True)
        break

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-struc_dir", type=str, required=True,)
parser.add_argument("-out_dir", type=str, required=True,)
parser.add_argument("-fluid_path", type=str, required=True,)
parser.add_argument("-sf_vdw", type=float, required=True,)

args = parser.parse_args()
struc_dir = args.struc_dir
out_dir = args.out_dir
fluid_path = args.fluid_path
sf_vdw = args.sf_vdw

vdw_radii = vdw_radii.copy()

ads = System.from_file(fluid_path)
ads.detect_bonds()


#  create_bulk_fluid(max_n_ads=100)
ins_fluid(max_n_ads=200, pbc=True)


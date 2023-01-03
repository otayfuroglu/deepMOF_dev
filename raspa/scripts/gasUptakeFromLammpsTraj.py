#! /truba/home/otayfuroglu/miniconda3/bin/python


#  from schnetpack.interfaces import SpkCalculator
#  from schnetpack.utils import load_model
#  from schnetpack import Properties
#  from schnetpack.environment import AseEnvironmentProvider
#
import RASPA
#  from RASPA2.raspa2 import *
#  from RASPA2.output_parser import parse

#  import openbabel
from openbabel import pybel
import pandas as pd
#  import sys, os

import multiprocessing
import multiprocessing.pool
# from itertools import product
#  import collections

from ase.io import read, write
#  import tqdm


def get_helium_void_fraction(cif_file_path):
    print("Helium void fraction is calculating...")
    structure = next(pybel.readfile("cif", cif_file_path))
    result = RASPA.get_helium_void_fraction(structure,
                                    cycles=1000,
                                    unit_cells=(1,1,1),
                                    #  forcefield="GenericMOFs",
                                    forcefield="CrystalGenerator",
                                    input_file_type="cif",
                                   )
    print("Done")
    return result


def getUptake(cif_file_path, pressure, helium_void_fraction):
    print(f"{molecule} uptake is calculating at {pressure} bar ...")
    structure = next(pybel.readfile("cif", cif_file_path))

    result = RASPA.run(
        structure, molecule,
        simulation_type="MonteCarlo",
        temperature=temperature, # in Kelvin
        pressure=pressure*1e5, # in Pascal
        helium_void_fraction=helium_void_fraction,
        unit_cells=(1,1,1),
        #  unit_cells=(2,2,2),
        framework_name="streamed", # if not streaming, this will load the structure at `$RASPA_DIR/share/raspa/structures`.
        cycles=1000,
        #  init_cycles="auto",
        init_cycles=500,
        #  forcefield="GenericMOFs",
        forcefield="CrystalGenerator",
        input_file_type="cif")

    uptake_grav_abs = result["Number of molecules"][molecule]\
                    ["Average loading absolute [milligram/gram framework]"][0]
    uptake_vol_abs = result["Number of molecules"][molecule]\
                    ["Average loading absolute [cm^3 (STP)/cm^3 framework]"][0]
    uptake_grav_excess = result["Number of molecules"][molecule]\
                    ["Average loading excess [milligram/gram framework]"][0]
    uptake_vol_excess = result["Number of molecules"][molecule]\
                    ["Average loading excess [cm^3 (STP)/cm^3 framework]"][0]

    #  print(uptake)
    print(f"Done for {pressure} bar ...")
    return (uptake_grav_abs, uptake_vol_abs,
            uptake_grav_excess, uptake_vol_excess,)


def prun(pressure):
    uptakes = getUptake(cif_file_path, pressure, helium_void_fraction)

    # order
    # uptake_grav_abs, uptake_vol_abs,
    # uptake_grav_excess, uptake_vol_excess,
    # uptake_grav_total, uptake_vol_total

    df[labels[0]] = [pressure]
    df[labels[1]] = [uptakes[0]]
    df[labels[2]] = [uptakes[1]]
    df[labels[3]] = [uptakes[2]]
    df[labels[4]] = [uptakes[3]]
    df.to_csv(csv_file_path, mode="a", header=False, index=False)

#  prun(0)

#for correction AssertionError: daemonic processes are not allowed to have children
# tested in python3.6
class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


if "__main__" == __name__:
    import argparse

    BASE_DIR = "/truba_scratch/otayfuroglu/deepMOF_dev/"
    mof_num = "1"
    temperature = 77
    descrip_word = "CrytalGen_NNPStructure_2x2x2_mostFreqConf"
    molecule = "H2"
    labels = [
        "Pressure", "AbsoluteUptake (mg/g)", "AbsoluteUptakes (cm^3 (STP)/cm^3)",
        "ExcessUptakes (mg/g)", "ExcessUptakes (cm^3 (STP)/cm^3)",

             ]

    csv_file_path = "IRMOF%s_%s_uptakes_%sK_%s.csv" % (
        mof_num, molecule, temperature, descrip_word)

    df = pd.DataFrame(columns=labels)
    df.to_csv(csv_file_path, index=False)

    #  traj_path = f"{BASE_DIR}/n2p2/works/runMD/latticeBulkModulusWithNNP/IRMOF{mof_num}_1Bar_{temperature+2}K.lammpstrj"
    #  traj_path = f"{BASE_DIR}/n2p2/works/runMD/latticeBulkModulusWithClassic/IRMOF{mof_num}_1Bar_{temperature+2}K.lammpstrj"
    #  ase_traj = read(traj_path, format="lammps-dump-text", index=":")
    #  init_vol = ase_traj[0].get_volume()
    idx = 750
    #  atoms = ase_traj[idx]
    #  cif_file_path = f"IRMOF{mof_num}_1Bar_298K_i{idx}Classic.cif"
    cif_file_path = "./IRMOF1_1Bar_300K_represent.cif"
    #  write(cif_file_path, atoms)

    helium_void_fraction = get_helium_void_fraction(cif_file_path)

    pressures = [1, 10, 20, 35, 50, 65, 80, 100, 120, 150, 180]

    for pressure in pressures:
        prun(pressure)

    #  pool = MyPool(processes=len(pressures))

    #  result_list_tqdm = []
    #  # implementation of  multiprocessor in tqdm. Ref.https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
    #  for result in tqdm.tqdm(pool.imap_unordered(func=prun, iterable=pressures), total=len(pressures)):
    #      result_list_tqdm.append(result)

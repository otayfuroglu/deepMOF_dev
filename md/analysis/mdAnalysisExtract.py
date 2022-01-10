#! /truba/home/yzorlu/miniconda3/bin/python -u

from ase.io.trajectory import Trajectory
from ase.visualize import view

from ase import Atoms
from ase.io import write
from ase.io.trajectory import TrajectoryReader, TrajectoryWriter

from schnetpack.md.utils import HDF5Loader
from schnetpack.md.utils import MDUnits
from schnetpack.md.utils import PowerSpectrum, IRSpectrum
from schnetpack import Properties

import os
import numpy as np
import pandas as pd
import tqdm
from multiprocessing import Pool


def getTime(data):

    print("Extraction Time")
    # Get the time axis
    return (np.arange(data.entries) * data.time_step / MDUnits.fs2internal)  # in fs


def getEnergy(data):
    print("Extraction Energies")
    # Get potential energies and check the shape
    energies = data.get_property(properties[0])
    return energies


def getTemperature(data):
    print("Extraction temperature")
    # Read the temperature
    temperature = data.get_temperature()

    # Compute the cumulative mean
    #  temperature_mean = np.cumsum(temperature) / (np.arange(data.entries)+1)
    return temperature #, temperature_mean


def getVibSpectra(data, resolution=4096):
    print("Extraction vibrational freqencies (Power)")
    # Intialize the spectrum
    spectrum = PowerSpectrum(data, resolution)

    # Compute the spectrum for the first molecule (default)
    spectrum.compute_spectrum(molecule_idx=0)

    # Get frequencies and intensities
    freqencies, intensities = spectrum.get_spectrum()
    return freqencies, intensities


def getIRVibSpectra(data, resolution=4096):
    """Requires the dipole moments
        to be present in the HDF5 dataset
    """
    print("Extraction vibrational freqencies (IR)")
    # Intialize the spectrum
    spectrum = IRSpectrum(data, resolution)

    # Compute the spectrum for the first molecule (default)
    spectrum.compute_spectrum(molecule_idx=0)

    # Get frequencies and intensities
    freqencies, intensities = spectrum.get_spectrum()
    return freqencies, intensities


def getAseAtoms(data, indx):

    return Atoms(
        data.get_property(Properties.Z, atomistic=True),
        positions=data.get_positions()[indx] * 10.0,
        cell=data.get_property(Properties.cell)[indx] * 10.0,
        pbc=data.pbc[0],
    )  # *10.0 for the hdf5 scaling

#  def _getVolume(i):
#      dummy_cell = Atoms(cell=equilibrated_data.get_property(Properties.cell)[i] * 10.0)
#      #  print(dummy_cell.get_cell_lengths_and_angles())
#      return dummy_cell.cell.volume
#
#  def getVolumes(data):
#      print("Extraction Cell Volume")
#      len_entries = data.entries
#      volumes = []
#
#      # implementation of  multiprocessor in tqdm.
#      # Ref.https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
#      pool = Pool(processes=40)
#      for result in tqdm.tqdm(pool.imap_unordered(func=_getVolume, iterable=range(len_entries)), total=len_entries):
#          volumes.append(result)
#
#      # Compute the cumulative mean
#      volumes_mean = np.cumsum(volumes) / (np.arange(data.entries)+1)
#
#      return volumes, volumes_mean

def getVolumes(data):
    print("Extraction Cell Volume")
    cells = np.array(equilibrated_data.get_property(Properties.cell) * 10.0, dtype=np.float64)
    volumes = np.abs(cells.sum(axis=1).prod(axis=1))

    # Compute the cumulative mean
    #  volumes_mean = np.cumsum(volumes) / (np.arange(data.entries)+1)

    return volumes # , volumes_mean

for mof_num in [7]:
    for temp in range(150, 151, 25):
        print("IRMOF-%s" %mof_num)
        md_type = "md"
        BASE_DIR = "/truba_scratch/yzorlu/deepMOF/HDNNP/schnetpack"
        job_name = "mercury_IRMOF%d_%dK_%s_NPT_111cell_allNNP_without5" %(mof_num, temp, md_type)
        file_name = "mercury_IRMOF%d_%s.hdf5"  %(mof_num, md_type)
        sub_jobname = "tec_md"
        MD_DIR = BASE_DIR + "/runMD/" + sub_jobname + "/" + job_name
        skip_initial = 0

        log_file_path = os.path.join(MD_DIR, file_name)
        #  data = HDF5Loader(log_file_path)
        equilibrated_data = HDF5Loader(log_file_path, skip_initial=skip_initial)
        properties = ["energy", "forces"]


        df = pd.DataFrame()
        df["Time"] = getTime(equilibrated_data)
        df["Energy"] = getEnergy(equilibrated_data)

        # get mean of cumulative eneregies
        #  df["EnergyMean"] = df["Energy"].expanding().mean()
        df["Temperature"] = getTemperature(equilibrated_data)
        df["Volume"] = getVolumes(equilibrated_data)
        df.to_csv("%s/mdAnalysis/results/%s/timeEnergyTempVolume_%s.csv" % (BASE_DIR, sub_jobname, job_name))

        #  df = pd.DataFrame()
        #  df["Freqencies"], df["Intensities"] = getVibSpectra(equilibrated_data,resolution=8192)
        #  df.to_csv("%s/mdAnalysis/results/%s/freqencyIntensityPower_%s.csv" % (BASE_DIR, sub_jobname, job_name))


        #  df = pd.DataFrame()
        #  df["Freqencies"], df["Intensities"] = getIRVibSpectra(equilibrated_data)
        #  df.to_csv("./results/IRMOF4/freqencyIntensityIR.csv")


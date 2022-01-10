#! /truba/home/yzorlu/miniconda3/envs/python38/bin/python -u

from ase.io.trajectory import Trajectory
from ase.visualize import view

from ase import Atoms
from ase.io import write
from ase.io.trajectory import TrajectoryReader, TrajectoryWriter

from schnetpack.md.utils import HDF5Loader
import matplotlib.pyplot as plt
from schnetpack.md.utils import MDUnits
from schnetpack.md.utils import PowerSpectrum
from schnetpack import Properties

import os, io
import numpy as np
import shutil

md_workdir = "../runMD/mof5_mercury_md_273Krpmd"
log_file = os.path.join(md_workdir, 'rpmd_mof5_mercury.hdf5')
data = HDF5Loader(log_file)
properties = ["energy", "forces"]


def plot_energy(data):
    # Get potential energies and check the shape
    energies = data.get_property(properties[0])
    print('Shape:', energies.shape)

    # Get the time axis
    time_axis = np.arange(data.entries)*data.time_step / MDUnits.fs2internal # in fs

    # Plot the energies 
    plt.figure()
    plt.plot(time_axis, energies)
    plt.ylabel('E [kcal/mol]')
    plt.xlabel('t [fs]')
    plt.tight_layout()
    plt.savefig("enegy.png")
    #  plt.show()
plot_energy(data)

def plot_temperature(data):
     # Read the temperature
     temperature = data.get_temperature()

     # Compute the cumulative mean
     temperature_mean = np.cumsum(temperature) / (np.arange(data.entries)+1)
     # Get the time axis
     time_axis = np.arange(data.entries)*data.time_step / MDUnits.fs2atu # in fs

     plt.figure(figsize=(8,4))
     plt.plot(time_axis, temperature, label='T')
     plt.plot(time_axis, temperature_mean, label='T (avg.)')
     plt.ylabel('T [K]')
     plt.xlabel('t [fs]')
     plt.legend()
     plt.tight_layout()
     plt.show()

#equilibrated_data = HDF5Loader(log_file, skip_initial=5000)
#plot_temperature(equilibrated_data)

def plot_vib_spectra(equilibrated_data):
    # Intialize the spectrum
    spectrum = PowerSpectrum(equilibrated_data, resolution=4096)

    # Compute the spectrum for the first molecule (default)
    spectrum.compute_spectrum(molecule_idx=0)

    # Get frequencies and intensities
    freqencies, intensities = spectrum.get_spectrum()

    # Plot the spectrum
    plt.figure()
    plt.plot(freqencies, intensities)
    plt.xlim(0,4000)
    plt.ylim(0,100)
    plt.ylabel('I [a.u.]')
    plt.xlabel('$\omega$ [cm$^{-1}$]')
    plt.show()

#plot_vib_spectra(equilibrated_data)

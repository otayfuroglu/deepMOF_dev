#! /truba/home/yzorlu/miniconda3/bin/python

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

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_context("paper", rc={"grid.linewidth": 0.8})
sns.set_style("ticks", rc={"grid.linestyle": "--"})

md_type = "md"
BASE_DIR = "/truba_scratch/yzorlu/deepMOF/HDNNP/schnetpack"
maker_types = ["v", "8", "s", "o", "x"]
colors = ["b", "m", "y", "g", "c"]
initial_skip = int(4e5)

def plot_energy(df):
    time_axis = df["Time"]
    energies = df["Energy"]
    #  energies_mean = df["EnergyMean"]

    # Compute the cumulative mean
    energies_mean = df["Energy"].expanding().mean() # with pandas

    #  Plot the energies
    plt.figure()
    plt.plot(time_axis, energies, c=colors[idx], label="E")
    plt.plot(time_axis, energies_mean, c="tab:orange", label='E (avg.)')
    plt.ylabel('E [kcal/mol]')
    plt.xlabel('t [fs]')
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/energies_%s.png" % (RESULTS_DIR, job_name), db=600)
    #  plt.show()


def plot_temperature(df):
    time_axis = df["Time"]
    temperature = df["Temperature"]
    #  temperature_mean = df["TemperatureMean"]

    # Compute the cumulative mean
    #  temperature_mean = np.cumsum(temperature) / (np.arange(df.entries)+1)
    temperature_mean = df["Temperature"].expanding().mean() # with pandas


    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, temperature, c=colors[idx], label='T')
    plt.plot(time_axis, temperature_mean, c="tab:orange", label='T (avg.)')
    plt.ylabel('T [K]')
    plt.xlabel('t [fs]')
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/temperature_%s.png" % (RESULTS_DIR, job_name), db=600)
    #  plt.show()


def plot_volume(df):
    time_axis = df["Time"]
    volume = df["Volume"]
    #  temperature_mean = df["VolumesMean"]

    # Compute the cumulative mean
    #  volumes_mean = np.cumsum(volumes) / (np.arange(df.entries)+1) # with numpy
    volumes_mean = df["Volume"].expanding().mean() # with pandas

    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, volume, c=colors[idx], label='V')
    plt.plot(time_axis, volumes_mean, c="tab:orange", label='V (avg.)')
    plt.ylabel('V [A^3]')
    plt.xlabel('t [fs]')
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/volume_%s.png" % (RESULTS_DIR, job_name), db=600)
    #  plt.show()

def plot_vib_spectra(df):
    freqencies = df["Freqencies"]
    intensities = df["Intensities"]

    # Plot the spectrum
    plt.figure()
    plt.plot(freqencies, intensities, c=colors[idx])
    plt.xlim(0, 4000)
    plt.ylim(0, 100)
    plt.ylabel('I [a.u.]')
    plt.xlabel('$\omega$ [cm$^{-1}$]')
    plt.savefig("%s/power_spectrum_%s.png" % (RESULTS_DIR, job_name), db=600)
    #  plt.show()

for idx, mof_num in enumerate([4]):

    job_name = "mercury_IRMOF%d_300K_%s_NPT_111cell_allNNP_without5" %(mof_num, md_type)
    sub_jobname = ""
    RESULTS_DIR = BASE_DIR + "/mdAnalysis/results/%s" %(sub_jobname)
    properties = ["energy", "forces"]

    df_energies = pd.read_csv("%s/timeEnergyTempVolume_%s.csv" % (RESULTS_DIR, job_name), skiprows=range(1, initial_skip))
    print(len(df_energies))
    plot_energy(df_energies)
    plot_temperature(df_energies)
    plot_volume(df_energies)
    #  df_power_spectrum = pd.read_csv("%s/freqencyIntensityPower_%s.csv" % (RESULTS_DIR, job_name))
    #  plot_vib_spectra(df_power_spectrum)

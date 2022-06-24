#
import lammps_logfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

import matplotlib
matplotlib.use("Agg")

sns.set_context("paper", rc={"grid.linewidth": 0.8})
sns.set_style("ticks", rc={"grid.linestyle": "--"})


import argparse

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-log", type=str, required=True)
parser.add_argument("-skip", type=int, required=True)
args = parser.parse_args()

colors = ["k",  "b", "midnightblue", "darkred", "firebrick", "b", "r", "dimgray", "orange", "m", "y", "g", "c"]


def plot_energy(data):
    # Get potential energies and check the shape
    labels = ["Pot. Energy", "Kin. Energy"]
    p_energies = data["PotEng"]
    k_energies = data["KinEng"]

    # Get the time axis
    time_axis = data["Step"]

    #  plt.show()
    #  Plot the energies
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(time_axis, p_energies, "-", c=colors[0], label=r"${%s}$"%labels[0])
    ax2.plot(time_axis, k_energies, "-",  c=colors[1], label=r"${%s}$"%labels[1])

    ax1.set_ylabel(r"Pot. Energy $(eV)$")
    ax2.set_ylabel(r"Kin. Energy $(eV)$")
    ax1.set_xlabel(r"Time $(fs)$")

    #  ax1.set_ylim(25, 80)
    #  ax2.set_ylim(350, 550)
    ax2.yaxis.label.set_color(colors[1])
    ax2.tick_params(axis='y', colors=colors[1])
    ax2.spines["right"].set_edgecolor(colors[1])

    ax1.legend(loc="upper center", prop={'size': 8.5}, bbox_to_anchor=(0.41, 0.46),
              fancybox=False, shadow=False, labelcolor=colors[0], frameon=False)
    ax2.legend(loc="upper center", prop={'size': 8.5}, bbox_to_anchor=(0.41, 0.40),
              fancybox=False, shadow=False, labelcolor=colors[1], frameon=False)

    plt.savefig("%s_energy.png" %log_base)

def plot_temperature(data):
    # Read the temperature
    temperature = data["Temp"]

    # Compute the cumulative mean
    temperature_mean = np.cumsum(temperature) / (np.arange(len(temperature))+1)
    # Get the time axis
    time_axis = data["Step"]

    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, temperature, label='T')
    plt.plot(time_axis, temperature_mean, label='T (avg.)')
    plt.ylabel('T (K)')
    plt.xlabel('Time (fs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s_temp.png" %log_base)
    #  plt.show()


def plot_loading(data, n_frame_atoms):
    # Read the loading
    n_loading = data["Atoms"].subtract(n_frame_atoms)

    # Compute the cumulative mean
    n_loading_mean = np.cumsum(n_loading) / (np.arange(len(n_loading))+1)
    # Get the time axis
    time_axis = data["Step"]

    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, n_loading, label='Number of Molecule')
    plt.plot(time_axis, n_loading_mean, label='Number of Molecule (avg.)')
    plt.ylabel('Number of Molecule')
    plt.xlabel('Time (fs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s_loading.png" %log_base)
    #  plt.show()


def plot_volume(data):
    # Read the volume
    volume = data["Volume"]

    # Compute the cumulative mean
    volume_mean = np.cumsum(volume) / (np.arange(len(volume))+1)
    # Get the time axis
    time_axis = data["Step"]

    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, volume, label=r"Volume")
    plt.plot(time_axis, volume_mean, label=r"Volume (avg.)")
    plt.ylabel(r"Volume ($\mathring{A}^3$)")
    plt.xlabel('t (fs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s_volume.png" %log_base)
    #  plt.show()


log_path = args.log
log_base = os.path.basename(log_path).split(".")[0]
log = lammps_logfile.File(log_path)

data = pd.DataFrame()
for label in log.get_keywords():
    data[label] = log.get(label)
#  df.to_csv("test.csv")

plot_energy(data)
n_frame_atoms = data["Atoms"][0]
initial_skip = args.skip
#  print(len(data))
data = data[initial_skip:]
plot_loading(data, n_frame_atoms)
plot_temperature(data)
plot_volume(data)

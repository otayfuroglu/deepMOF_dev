#
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from ase.io.trajectory import Trajectory
from ase.io import read
from ase.units import bar, GPa
import argparse



import matplotlib
matplotlib.use("Agg")

sns.set_context("paper", rc={"grid.linewidth": 0.8})
sns.set_style("ticks", rc={"grid.linestyle": "--"})



def plot_pot_energy():

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    # for classic numbering format
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)

    ax.plot(time_axis, pot_energies, label='Energy')
    #  ax.plot(time_axis, po_energy_mean, label='Energy (avg.)')
    ax.set_ylabel('Energy (eV)')
    ax.set_xlabel('Time (fs)')
    ax.legend()
    #  plt.tight_layout()
    plt.savefig("%s_pot_energy.png" %log_base)
    #  plt.show()


def plot_energy():
    # Get potential energies and check the shape
    labels = ["Pot. Energy", "Kin. Energy"]

    #  plt.show()
    #  Plot the energies
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(time_axis, pot_energies, "-", c=colors[0], label=r"${%s}$"%labels[0])
    ax2.plot(time_axis, kin_energies, "-",  c=colors[1], label=r"${%s}$"%labels[1])

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

def plot_temperature():

    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, temps, label='T')
    plt.ylabel('T (K)')
    plt.xlabel('Time (fs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s_temp.png" %log_base)
    #  plt.show()


def plot_volume():

    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, vols, label=r"Volume")
    #  plt.plot(time_axis, volume_mean, label=r"Volume (avg.)")
    plt.ylabel(r"Volume ($\mathring{A}^3$)")
    plt.xlabel('t (fs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s_volume.png" %log_base)
    #  plt.show()

def plot_pressure():

    #  window = 10
    #  new_length = (len(pressures) // window) * window
    #  pressures_mean = np.mean(np.array(pressures[:new_length]).reshape(-1, window), axis=1)   # moving avarage
    #  pressures_mean = pressures_mean.repeat(window) # for same len of time axis

    pressures_mean = np.cumsum(pressures) / (np.arange(len(pressures))+1) # cumulative avarage

    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, pressures, label=r"Pressure")
    plt.plot(time_axis[:len(pressures_mean)], pressures_mean, label=r"Pressure (avg.)")
    plt.ylabel(r"Pressure (bar)")
    plt.xlabel('t (fs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s_pressure.png" %log_base)
    #  plt.show()

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-traj", type=str, required=True)
parser.add_argument("-stepsize", type=float, required=True)
parser.add_argument("-skip", type=int, required=True)
args = parser.parse_args()

colors = ["k",  "b", "midnightblue", "darkred", "firebrick", "b", "r", "dimgray", "orange", "m", "y", "g", "c"]


traj_path = args.traj
initial_skip = args.skip
log_base = os.path.basename(traj_path).split(".")[0]

atoms_list = read(traj_path, format="lammps-dump-text", index=":", parallel=True)

time_axis = np.arange(len(atoms_list[initial_skip:])) * args.stepsize
temps = [atoms.get_temperature() for atoms in atoms_list[initial_skip:]]
#  pot_energies = [atoms.get_potential_energy() for atoms in atoms_list[initial_skip:]]
#  kin_energies = [atoms.get_kinetic_energy() for atoms in atoms_list[initial_skip:]]
vols = [atoms.get_volume() for atoms in atoms_list[initial_skip:]]
cell_lengths = [atoms.cell.lengths() for atoms in atoms_list[initial_skip:]]

#  pressures = [(-atoms.get_stress(voigt=False, include_ideal_gas=True).trace() / 3) / bar # eV/A^3 to bar
             #  for atoms in atoms_list[initial_skip:]]


#  print(" Avg. T (K): ", temps.mean())
print(" Avg. Volume (A^3): ", np.array(vols).mean())
print(" Avg. cell lengths a, b, c ", np.array(cell_lengths).mean(axis=0))

#  plot_energy()
#  plot_pot_energy()
#  plot_loading(, n_frame_atoms)

plot_temperature()
plot_volume()
#  plot_pressure()

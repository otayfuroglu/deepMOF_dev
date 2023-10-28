
from ase import Atoms
from ase.io import read, write
from ase.geometry.analysis import Analysis

from matplotlib import pyplot as plt
import numpy as np
import argparse

import seaborn as sns
sns.set_context("paper", rc={"grid.linewidth": 0.8})
sns.set_style("ticks", rc={"grid.linestyle": "--"})

plt.figure(figsize=(10, 5))


def lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair):
    symbols = [atom_type_symbol_pair[key] for key in lammps_atoms.get_atomic_numbers()]
    atoms = Atoms(symbols=symbols, positions=lammps_atoms.positions, cell=lammps_atoms.cell)
    atoms.pbc = True
    return atoms


def plot_loading(num_gases_list):
    # Read the loading
    n_loading = num_gases_list

    # Compute the cumulative mean
    #  n_loading_mean = np.cumsum(n_loading) / (np.arange(len(n_loading))+1)
    # Get the time axis
    time_axis = np.arange(len(num_gases_list)) * interval * step_size

    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, n_loading, label='Number of Molecule')
    #  plt.plot(time_axis, n_loading_mean, label='Number of Molecule (avg.)')
    plt.ylabel('Number of Molecule')
    plt.xlabel('Time (fs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s_loading.png" %log_base)


    #  plt.show()
#  parser = argparse.ArgumentParser(description="Give something ...")
#  parser.add_argument("-trj_path", type=str, required=True, help="..")
#  args = parser.parse_args()


#  index = 3000
#  elements_list = [("Zn", "Zn"), ("Zn", "O"), ("Zn", "C"), ("O", "O")]
#  elements_list = [("Zn", "Ne"), ("C", "Ne"), ("O", "Ne")]
elements_list = [("O", "C")]

#  elements = ("Zn", "Zn")
index = slice(0, 3900, 1)
#  index = slice(0, -1, 1)

a1, a2, a3, a4 = 4703, 4798, 4845, 4892
b1, b2 = 4665, 4703
c1, c2, c3 = 4703, 4798, 4737

#  atom_type_symbol_pair = {1:"Zn", 2:"O", 3:"O", 4:"C", 5:"H", 6:"Ne"}
atom_type_symbol_pair = {1:"Zn", 2:"O", 3:"C", 4:"H", 5:"Ne"}

step_size = 0.5
interval = 1000

mv_avg = 1
#
#  rdf = np.load(f"{'_'.join(elements)}_avg_rdf.npy")
#  #  rdf = np.load("frame_3000_Zn_Zn_rdf.npy")
#
#  dist_list = np.linspace(0.0, rmax, nbins)
#  plt.plot(dist_list, rdf.squeeze())
#  plt.savefig(f"frame_{index}_{'_'.join(elements)}.rdf.png")
#  plt.show()
#
#  quit()

#  lammps_trj_path = args.trj_path


BASE_DIR = "/truba_scratch/otayfuroglu/deepMOF_dev/n2p2/works/runMD"

properties_mean = {}
for prefix in ["", "classic"]:
    lammps_trj_path = f"{BASE_DIR}/{prefix}flexAdorpsionH2onIRMOF10inNPT_77K/IRMOF10_H2_100Bar_77K.lammpstrj"
    #  lammps_trj_path = f"{BASE_DIR}/{prefix}FlexAdorpsionCH4onIRMOF10inNPT_300K/test.lammpstrj"

    if prefix == "classic":
        atom_type_symbol_pair = {1:"Zn", 2:"O", 3:"O", 4:"C", 5:"H", 6:"Ne"}
        prefix = "UFF"
        print("\n", prefix)
    else:
        prefix = "HDNNP"
        print("\n", prefix)

    lammps_atoms_list = read(lammps_trj_path, format="lammps-dump-text", index=index, parallel=True)
    dihedeals = []
    distances = []
    angles = []

    n_gases = []
    for lammps_atoms in lammps_atoms_list:
        atoms = lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair)
        dihedeals += [atoms.get_dihedral(a1, a2, a3, a4, mic=True)]
        distances += [atoms.get_distance(b1, b2, mic=True)]
        #  if atoms.get_distance(b1, b2) > 5:
            #  write("broken.extxyz", atoms)
        angles += [atoms.get_angle(c1, c2, c3, mic=True)]
        n_gases += [len(atoms[atoms.numbers == 10])]
#  avg_rdf = np.mean(rdfs, axis=0)
#  np.save(f"{'_'.join(elements)}_avg_rdf", avg_rdf)


    # cumulative avarage
    #  properties_mean[f"dihedeals_{prefix}"] = np.cumsum(np.array(dihedeals)) / (np.arange(len(dihedeals))+1)
    #  properties_mean[f"distances_{prefix}"] = np.cumsum(np.array(distances)) / (np.arange(len(distances))+1)
    #  properties_mean[f"angles_{prefix}"] = np.cumsum(np.array(angles)) / (np.arange(len(angles))+1)

    # moving avarage
    properties_mean[f"dihedeals_{prefix}"] = np.mean(np.array(dihedeals).reshape(-1, mv_avg), axis=1)
    properties_mean[f"distances_{prefix}"] = np.mean(np.array(distances).reshape(-1, mv_avg), axis=1)
    properties_mean[f"angles_{prefix}"] = np.mean(np.array(angles).reshape(-1, mv_avg), axis=1)
    properties_mean[f"n_gases_{prefix}"] = np.mean(np.array(n_gases).reshape(-1, mv_avg), axis=1)

import pandas as pd

df = pd.DataFrame()
for item in ["dihedeals", "distances", "angles", "n_gases"]:
    #  fig = plt.figure()
    #  ax = fig.add_subplot(111)
    for key in [key for key in properties_mean.keys() if item in key]:
        plt.plot(np.arange(len(properties_mean[key])) * step_size * interval * mv_avg, # for time as x axis
                 properties_mean[key], label=f"{key.split('_')[-1]}")
        df[key] = properties_mean[key]
     #     plt.plot(np.arange(len(properties_mean[key])) * step_size * interval * mv_avg, # for time as x axis
     #              properties_mean[key])
        #  plt.plot(properties_mean[f"n_gases_{key.split('_')[-1]}"], properties_mean[key], label=f"{key.split('_')[-1]}")
        plt.xlabel(r"Time (fs)")
        if item == "distances":
            plt.ylabel(r" Distance ($\AA$)")
        elif item == "n_gases":
            plt.ylabel(r"Number of Molecules")
        else:
            plt.ylabel(r" Angle ($^0$)")
        #  plt.show()
    plt.legend()
    plt.savefig(f"mv_avg_{mv_avg}_{item}.png")
    plt.clf()
    #  plt.savefig(f"{'_'.join((str(a1), str(a2), str(a3), str(a4)))}_dihedral.png")

df.to_csv("data.csv")
#  plt.plot(rdf[:,1,:].squeeze(), rdf[:,0,:].squeeze())
#  plt.savefig(f"{'_'.join(elements)}_avg_rdf.png")
#  plt.show()



from ase import Atoms
from ase.io import read, write
from ase.geometry.analysis import Analysis

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
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



#  parser = argparse.ArgumentParser(description="Give something ...")
#  parser.add_argument("-trj_path", type=str, required=True, help="..")
#  args = parser.parse_args()


#  index = 3000
#  elements_list = [("Zn", "Zn"), ("Zn", "O"), ("Zn", "C"), ("O", "O")]
#  elements_list = [("Zn", "Ne"), ("C", "Ne"), ("O", "Ne")]
elements_list = [("Zn", "Zn")]

#  elements = ("Zn", "Zn")
#  index = slice(3000, -1, 100)
start = 0
end = 100
index = slice(start, end, 1)
rmax, nbins = 10.0, 200

#  atom_type_symbol_pair = {1:"Zn", 2:"O", 3:"O", 4:"C", 5:"H", 6:"Ne"}
atom_type_symbol_pair = {1:"Zn", 2:"O", 3:"C", 4:"H", 5:"Ne"}
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

dist_list = np.linspace(0.0, rmax, nbins)
rdfs = []
rdfs_particle_pair = {}


BASE_DIR = "/Users/omert/Desktop/deepMOF_dev/n2p2/works/runMD/"
for prefix in ["", "classic"]:
    lammps_trj_path = f"{BASE_DIR}/{prefix}FlexAdorpsionCH4onIRMOF10inNPT_300K/IRMOF10_CH4_100Bar_300K.lammpstrj"
    print(lammps_trj_path)

    if prefix == "classic":
        atom_type_symbol_pair = {1:"Zn", 2:"O", 3:"O", 4:"C", 5:"H", 6:"Ne"}
        prefix = "UFF"
        print("\n", prefix)
    else:
        prefix = "HDNNP"
        print("\n", prefix)

    for lammps_atoms in read(lammps_trj_path, format="lammps-dump-text", index=index, parallel=True):
        atoms = lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair)
        #  write(f"frame_{index}.extxyz", atoms)
        analysis = Analysis(atoms)
        for elements in elements_list:
            #  print(elements)
            result = analysis.get_rdf(rmax=rmax, nbins=nbins, elements=elements)
            rdfs_particle_pair[f"{prefix}_{'_'.join(elements)}"] = result

#  avg_rdf = np.mean(rdfs, axis=0)
#  np.save(f"{'_'.join(elements)}_avg_rdf", avg_rdf)


df = pd.DataFrame()

for particle_pair, rdfs in rdfs_particle_pair.items():
    y_data = np.mean(rdfs, axis=0).squeeze()
    plt.plot(dist_list, y_data, label=particle_pair)
    df[particle_pair] = y_data

df.to_csv(f"Frame{start}_{end}_{'_'.join([''.join(elements) for elements in elements_list])}.csv")

#  plt.legend()
#  plt.xlabel(r"r [$\AA$]")
#  plt.ylabel(r"g(r)")
#  #  plt.show()
#  plt.savefig(f"Frame{start}_{end}_{'_'.join([''.join(elements) for elements in elements_list])}")




#  plt.plot(rdf[:,1,:].squeeze(), rdf[:,0,:].squeeze())
#  plt.savefig(f"{'_'.join(elements)}_avg_rdf.png")
#  plt.show()

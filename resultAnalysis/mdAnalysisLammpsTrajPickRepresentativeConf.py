from ase.io import read, write
import argparse
import os

from openbabel import openbabel

#  from  rdkit import Chem
#  import os_util
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
#  from scipy.cluster.vq import kmeans, vq, whiten

import multiprocessing
import multiprocessing.pool
from itertools import product

import numpy as np
import shutil
import yaml
from collections import ChainMap
#  import tqdm


def loadMolWithOB(mol_path):

    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat(parameters["outformat"])
    ob_mol = openbabel.OBMol()
    obConversion.ReadFile(ob_mol, mol_path)

    return ob_mol


def getRmsdOBMol(mol1, mol2):

    obAlign = openbabel.OBAlign(mol1, mol2)
    obAlign.Align()
    return obAlign.GetRMSD()


def getMolListOB(conf_dir):
    supp_dict = {loadMolWithOB(f"{conf_dir}/{fl_name}"):
                  fl_name for fl_name in os.listdir(conf_dir)
                  if fl_name.endswith(parameters["outformat"])}
    mol_list = []
    for mol, fl_name in supp_dict.items():
        mol.SetTitle(fl_name)
        mol_list.append(mol)

    return mol_list


def calc_rmsdWithOB(i, j):
    # calc RMSD
    return i, j, getRmsdOBMol(mol_list[i], mol_list[j])


def getClusterRMSDFromFiles(conf_dir, n_processes=100):

    print("Loading Molecules ...")
    global mol_list
    mol_list = getMolListOB(conf_dir)
    #  mol_list = getMolListRD(conf_dir)

    n_mol=len(mol_list)
    print("Number Molecules: ", n_mol)
    if n_mol <= 1:
        print("Clustering do not applied.. There is just one conformer")
        return 0

    print("Calculating pair distance matrix ...")
    with  multiprocessing.Pool(processes=n_processes) as pool:
        results = pool.starmap(calc_rmsdWithOB, product(range(n_mol), repeat=2))

    dist_matrix = np.empty(shape=(n_mol, n_mol))
    for result in results:
        dist_matrix[result[0], result[1]] = result[2]
    #  print(dist_matrix[0][1])

    print("Clsutering process...")
    linked = linkage(dist_matrix,'complete')

    cluster_conf = defaultdict(list)
    labelList = [mol.GetTitle() for mol in mol_list]
    #  for key, fl_name in zip(cluster, labelList):
    for key, fl_name in zip(fcluster(linked, t=parameters["rmsd_thresh"], criterion='distance'), labelList):
        cluster_conf[key].append(fl_name)

        #  to place clustured files seperately
        directory = f"{conf_dir}/cluster_{key}"
        if not os.path.exists(directory):
            os.mkdir(directory)
        os.replace(f"{conf_dir}/{fl_name}", f"{directory}/{fl_name}")

    fl = open("removeFileNamesFromDB_%s.csv" %conf_dir.replace("/", ""), "w")
    fl.write("FileNames\n")

    select_dir = f"{conf_dir}/selected"
    if not os.path.exists(select_dir):
        os.mkdir(select_dir)

    for key, fl_names in cluster_conf.items():
        if len(fl_names) == 0:
            print("Empty")
            continue

        directory = f"{conf_dir}/cluster_{key}"
        if not os.path.exists(directory):
            os.mkdir(directory)

        for i, fl_name in enumerate(fl_names):
            if i == 0:
                # copy selected file to selected dir
                shutil.copyfile(f"{conf_dir}/{fl_name}", f"{select_dir}/{fl_name}")
            else:
                # sava to csv for remove
                fl.write(fl_name.split(".")[0]+"\n")

            # to place clustured files seperately
            os.replace(f"{conf_dir}/{fl_name}", f"{directory}/{fl_name}")


def lammpsTrajSperate():
    print("Reding lammps traj...")
    ase_traj = read(parameters["trajpath"], format="lammps-dump-text", index=":")
    print(parameters["skip"], len(ase_traj))
    print("Obtainig conformers from lammps traj...")
    for idx in range(0, len(ase_traj), parameters["interval"]):
        conf_path = f"{conf_dir}/conf_{idx}.{parameters['outformat']}"
        atoms = ase_traj[idx]
        # cornvert from Lammps atom type to ase atomic numbers
        atom_nums = atoms.get_atomic_numbers()
        for i, lammps_atom_num in enumerate(atom_nums):
            for atom_num_key in atomnum_pairs.keys():
                if lammps_atom_num == int(atom_num_key):
                    #  print(lammps_atom_num, atomnum_pairs[atom_num_key])
                    atom_nums[i] = atomnum_pairs[atom_num_key]
                    break
        atoms.set_atomic_numbers(atom_nums)
        write(conf_path, atoms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Give something ...")
    parser.add_argument("-in", "--input", type=str)
    args = parser.parse_args()
    args = parser.parse_args()
    conf_file = args.input

    with open(conf_file) as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)

    atomnum_pairs = dict(ChainMap(*parameters["atomnum_pairs"])) # merge dicts which in list

    conf_dir = f"{parameters['outformat']}_confs"
    if not os.path.exists(conf_dir):
        os.mkdir(conf_dir)

    n_processes = 100

    lammpsTrajSperate()
    getClusterRMSDFromFiles(conf_dir, n_processes=100)

from openbabel import openbabel

from  rdkit import Chem
#  import os_util
from collections import defaultdict
#  from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.vq import kmeans, vq, whiten

import multiprocessing
from itertools import product

import numpy as np
import argparse
import os
import shutil


def loadMolWithOB(mol_path):

    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat("pdb")
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
                  if fl_name.endswith(".pdb")}
    mol_list = []
    for mol, fl_name in supp_dict.items():
        mol.SetTitle(fl_name)
        mol_list.append(mol)

    return mol_list


def calc_rmsdWithOB(i, j):
    # calc RMSD
    return i, j, getRmsdOBMol(mol_list[i], mol_list[j])


def calc_rmsdWithRD(i, j):
    # calc RMSD
    return i, j, Chem.rdMolAlign.GetBestRMS(mol_list[i], mol_list[j])


def getMolListRD(conf_dir):
    supp_dict = {Chem.MolFromPDBFile(f"{conf_dir}/{fl_name}"):
              fl_name for fl_name in os.listdir(conf_dir)
              if fl_name.endswith(".pdb")}
    mol_list = []
    for mol, fl_name in supp_dict.items():
        mol.SetProp("_Name", fl_name)
        mol_list.append(mol)

    return mol_list


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

    dist_matrix=np.empty(shape=(n_mol, n_mol))
    for result in results:
        dist_matrix[result[0], result[1]] = result[2]
    #  print(dist_matrix[0][1])

    print("Clsutering process...")
    n_group = int(n_mol * n_gruop_ratio)
    print("Nuber of cluster: ", n_group)
    whitened = whiten(dist_matrix)
    centroids, _ = kmeans(whitened, n_group)
    cluster, _ = vq(whitened,centroids)

    cluster_conf = defaultdict(list)
    labelList = [mol.GetTitle() for mol in mol_list]
    for key, fl_name in zip(cluster, labelList):
        cluster_conf[key].append(fl_name)

        # to place clustured files seperately
        #  directory = f"{conf_dir}/cluster_{key}"
        #  if not os.path.exists(directory):
        #      os.mkdir(directory)
        #  os.replace(f"{conf_dir}/{fl_name}", f"{directory}/{fl_name}")

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

parser = argparse.ArgumentParser(description="")
parser.add_argument("-confDIR", type=str, required=False)
parser.add_argument("-n_gruop_ratio", type=float, required=True)
parser.add_argument("-n_procs", type=int, required=True)

args = parser.parse_args()

#  conf_dir = "nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling_mof5_f1"
conf_dir = args.confDIR
n_processes = args.n_procs
n_gruop_ratio = args.n_gruop_ratio
getClusterRMSDFromFiles(conf_dir, n_processes)

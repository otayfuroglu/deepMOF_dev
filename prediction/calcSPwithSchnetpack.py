#
import torch
#  import numpy as np
#  from math import sqrt

from schnetpack.interfaces import SpkCalculator
from schnetpack.utils import load_model
from schnetpack import Properties
from schnetpack.datasets import AtomsData
from schnetpack.environment import AseEnvironmentProvider

import pandas as pd
#  from scipy import stats
#  from matplotlib import pyplot as plt

from multiprocessing import Pool

import tqdm
from ase.io import write, read

import numpy as np

import os, sys, warnings
index_warning = 'Converting sparse IndexedSlices'
warnings.filterwarnings('ignore', index_warning)

import argparse

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-val_type", "--val_type", type=str, required=True)
parser.add_argument("-MODEL_DIR", "--MODEL_DIR", type=str, required=True)
parser.add_argument("-RESULT_DIR", "--RESULT_DIR", type=str, required=True)
parser.add_argument("-filesDIR", "--filesDIR", type=str, required=False)
parser.add_argument("-db_path", "--db_path", type=str, required=False)
args = parser.parse_args()

#  mof_num = args.mof_num
mode = args.val_type
MODEL_DIR = args.MODEL_DIR
RESULT_DIR = args.RESULT_DIR
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


USER=RESULT_DIR.split("/")[2]
BASE_DIR = f"/truba_scratch/{USER}/deepMOF/HDNNP"

#  if mode == "train" or mode == "test":
#      path_to_db = ("%s/prepare_data/workingOnDataBase/"
#                    "nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling_IRMOFseries1_4_6_7_10_merged_50000_ev.db" %(BASE_DIR))
#      data = AtomsData(path_to_db)
#  elif mode == "torsion":
#      path_to_db = ("%s/prepare_data/dataBases/"
#                    "mof5_phenyl_torsion_ev_v2.db" %BASE_DIR)
#      data = AtomsData(path_to_db)
#  elif mode == "aromatic_ch_bond":
#      path_to_db = ("%s/prepare_data/dataBases/"
#                    "mof5_phenyl_CH_bond_ev.db" %BASE_DIR)
#      data = AtomsData(path_to_db)
#  elif mode == "aliphatic_ch_bond":
#      path_to_db = ("%s/prepare_data/dataBases/"
#                    "aliphaticCHBondEnergyForecesDMomentWithORCA_TZVP_fromScalingIRMOFseries4_ev.db" %BASE_DIR)
#    data = AtomsData(path_to_db)
if mode == "train" or mode == "test" or mode == "fromDB":
    db_path = args.db_path
    data = AtomsData(db_path)
elif mode == "fromFiles":
    filesDIR = args.filesDIR
    file_names = [file_name for file_name in os.listdir(filesDIR)]
    print("Working on xyz files which in ", filesDIR.split("/")[-1])
else:
    print("Error: Ivalid calculation type")
    sys.exit(1)



if mode == "fromFiles":
    column_names_energy = [
        "FileNames",
        "schnet_SP_energies",
        "schnet_SP_energiesPerAtom",
    ]

    column_names_fmax = [
        "FileNames"
        "schnet_SP_fmax",
    ]

    csv_file_name_energy = "qm_sch_SP_E_%s.csv" %(mode)
    csv_file_name_fmax = "qm_sch_SP_F_%s.csv" %(mode)

    df_data_energy = pd.DataFrame(columns=column_names_energy)
    df_data_fmax = pd.DataFrame(columns=column_names_fmax)

    df_data_energy.to_csv("%s/%s" %(RESULT_DIR, csv_file_name_energy), float_format='%.6f')
    df_data_fmax.to_csv("%s/%s"%(RESULT_DIR, csv_file_name_fmax), float_format='%.6f')


else:
    column_names_energy = [
        "FileNames",
        "qm_SP_energies",
        "schnet_SP_energies",
        "Error",
        "qm_SP_energiesPerAtom",
        "schnet_SP_energiesPerAtom",
        "ErrorPerAtom",
    ]

    column_names_fmax = [
        "FileNames",
        "qm_SP_fmax",
        "schnet_SP_fmax",
        "Error",
    ]

    column_names_fmax_component = [
        "FileNames",
        "qm_SP_fmax_component",
        "schnet_SP_fmax_component",
        "Error",
    ]

    csv_file_name_energy = "qm_sch_SP_E_%s.csv" %(mode)
    csv_file_name_fmax = "qm_sch_SP_F_%s.csv" %(mode)
    csv_file_name_fmax_component = "qm_sch_SP_FC_%s.csv" %(mode)

    df_data_energy = pd.DataFrame(columns=column_names_energy)
    df_data_fmax = pd.DataFrame(columns=column_names_fmax)
    df_data_fmax_component = pd.DataFrame(columns=column_names_fmax_component)

    df_data_energy.to_csv("%s/%s" %(RESULT_DIR, csv_file_name_energy), float_format='%.6f')
    df_data_fmax.to_csv("%s/%s"%(RESULT_DIR, csv_file_name_fmax), float_format='%.6f')
    df_data_fmax_component.to_csv("%s/%s" %(RESULT_DIR, csv_file_name_fmax_component), float_format='%.6f')


def get_trainValTestData_idx(mode):
    data_idx = np.load("%s/schnetpack/runTraining/%s/split.npz" %(BASE_DIR, MODEL_DIR))
    if mode == "train":
        return [int(i) for i in data_idx["train_idx"]]
    elif mode == "test":
        val_idx = [int(i) for i in data_idx["val_idx"]]
        return val_idx[:int(len(val_idx)/2)]
    else:
        print("Invalid validation type")
        sys.exit(1)


def get_fRMS(forces):
    """
    Args:
    forces(3D torch tensor)
    retrun:
    force RMS (zero D torch tensor)

     Example:
     A = tensor([[-0.000000154,-0.000003452,-0.000000532]
                 [-0.000001904,-0.000001209, 0.000000163]
                 [-0.000005395,-0.000002006,-0.000000011]
                 [ 0.000002936, 0.000000314, 0.000002471]
                 [ 0.000001738, 0.000008122,-0.000001215]
                 [ 0.000000513,-0.000006310,-0.000002488]
                 [ -0.000000256,-0.000004267, 0.00001445]
                 [ -0.000000579,-0.000001933,-0.00000231]
                 [ -0.000001118, 0.000001107,-0.00000152]])

    print("Cartsian Forces: RMS" , torch.sqrt(((A -A.mean(axis=0))**2).mean()) #RMS force
    Output:  Cartsian Forces: RMS     5.009549113310641e-06
    """

    if not torch.is_tensor(forces):
        forces = torch.from_numpy(forces) # if numpy array convert to 3d tensor
    else:
        forces = forces.squeeze(0)

    return torch.sqrt(((forces -forces.mean(axis=0))**2).mean())

def get_fmax_atomic(forces):
    """
    Args:
    forces(3D torch tensor)
    retrun:
    maximum atomic force (zero D torch tensor)

    Example:
     A = tensor([[-0.000000154,-0.000003452,-0.000000532]
                 [-0.000001904,-0.000001209, 0.000000163]
                 [-0.000005395,-0.000002006,-0.000000011]
                 [ 0.000002936, 0.000000314, 0.000002471]
                 [ 0.000001738, 0.000008122,-0.000001215]
                 [ 0.000000513,-0.000006310,-0.000002488]
                 [ -0.000000256,-0.000004267, 0.00001445]
                 [ -0.000000579,-0.000001933,-0.00000231]
                 [ -0.000001118, 0.000001107,-0.00000152]])

    print(("Cartsian Forces: Max atomic force", (A**2).sum(1).max()**5 # Maximum atomic force (fom ASE)
    Output: Cartsian Forces: Max atomic forces 0.0000
    """

    if not torch.is_tensor(forces):
        forces = torch.from_numpy(forces) # if numpy array convert to 3d tensor
    else:
        forces = forces.squeeze(0)

    return (forces**2).sum(1).max()**5 # Maximum atomic force (fom ASE)

def get_fmax_component(forces):
    """
    Args:
    forces(3D torch tensor)

    retrun:
    maximum force conponent(zero D torch tensor)

    Example:
     A = tensor([[-0.000000154,-0.000003452,-0.000000532]
                 [-0.000001904,-0.000001209, 0.000000163]
                 [-0.000005395,-0.000002006,-0.000000011]
                 [ 0.000002936, 0.000000314, 0.000002471]
                 [ 0.000001738, 0.000008122,-0.000001215]
                 [ 0.000000513,-0.000006310,-0.000002488]
                 [ -0.000000256,-0.000004267, 0.00001445]
                 [ -0.000000579,-0.000001933,-0.00000231]
                 [ -0.000001118, 0.000001107,-0.00000152]])

    print("Cartsian Forces: Max", get_fmax_component(A)
    Output:Cartsian Forces: Max -1.831199915613979e-05
    """

    if not torch.is_tensor(forces):
        forces = torch.from_numpy(forces) # if numpy array convert to 3d tensor
    else:
        forces = forces.squeeze(0)

    abs_forces = forces.abs()
    abs_idxs = (abs_forces==torch.max(abs_forces)).nonzero().squeeze(0) # get index of max value in abs_forces.
    if len(abs_idxs.shape) > 1: # if there are more than one max value
        abs_idxs = abs_idxs[0]  # select just one
    return forces[abs_idxs[0], abs_idxs[1]].item()

def get_fmax_idx(forces):
    """
    Args:
    forces(3D torch tensor)

    retrun:
    maximum force conponent indices (zero 1D torch tensor)
    """

    if not torch.is_tensor(forces):
        forces = torch.from_numpy(forces) # if numpy array convert to 3d tensor
    else:
        forces = forces.squeeze(0)

    abs_forces = forces.abs()
    abs_idxs = (abs_forces==torch.max(abs_forces)).nonzero().squeeze(0) # get index of max value in abs_forces.
    if len(abs_idxs.shape) > 1: # if there are more than one max value
        abs_idxs = abs_idxs[0]  # select just one
    return abs_idxs

def get_fmax_componentFrom_idx(forces, fmax_component_idx):
    """
    xxx
    """

    if not torch.is_tensor(forces):
        forces = torch.from_numpy(forces) # if numpy array convert to 3d tensor
    else:
        forces = forces.squeeze(0)

    return forces[fmax_component_idx[0], fmax_component_idx[1]].item()


def getSPEneryForces(idx):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx % 2)
    model_schnet = load_model("%s/schnetpack/runTraining/%s/best_model" %(BASE_DIR, MODEL_DIR),
                              map_location=device)

    calc_schnet = SpkCalculator(model_schnet, device=device,
                                energy=Properties.energy,
                                forces=Properties.forces,
                                #  collect_triples=True,
                                environment_provider=AseEnvironmentProvider(cutoff=6.0)
                               )

    file_names = [data.get_name(idx)]
    mol = data.get_atoms(idx)
    #  write("test_atom_%d.xyz" %idx, mol)
    n_atoms = mol.get_number_of_atoms()

    qm_energy = data[idx][Properties.energy]
    qm_fmax = data.get_fmax(idx)

    # first get fmax indices than get qm_fmax and schnet fmax component
    qm_fmax_component_idx = get_fmax_idx(data[idx][Properties.forces])
    qm_fmax_component = get_fmax_componentFrom_idx(data[idx][Properties.forces],
                                                   qm_fmax_component_idx)

    mol.set_calculator(calc_schnet)
    schnet_energy = mol.get_potential_energy()
    schnet_fmax = (mol.get_forces()**2).sum(1).max()**0.5  # Maximum atomic force (fom ASE).
    schnet_fmax_component = get_fmax_componentFrom_idx(mol.get_forces(),
                                                       qm_fmax_component_idx)

    energy_err = qm_energy - schnet_energy
    fmax_err = qm_fmax - schnet_fmax
    fmax_component_err = qm_fmax_component - schnet_fmax_component


    energy_values = [file_names,
                     qm_energy,
                     schnet_energy,
                     energy_err,
                     qm_energy/n_atoms,
                     schnet_energy/n_atoms,
                     energy_err/n_atoms,
                    ]

    fmax_values = [file_names,
                     qm_fmax,
                     schnet_fmax,
                     fmax_err,
                    ]

    fmax_component_values = [file_names,
                     qm_fmax_component,
                     schnet_fmax_component,
                     fmax_component_err,
                    ]

    for column_name, value in zip(column_names_energy, energy_values):
        df_data_energy[column_name] = value

    for column_name, value in zip(column_names_fmax, fmax_values):
        df_data_fmax[column_name] = value

    for column_name, value in zip(column_names_fmax_component, fmax_component_values):
        df_data_fmax_component[column_name] = value

    df_data_energy.to_csv("%s/%s" %(RESULT_DIR, csv_file_name_energy), mode="a", header=False, float_format='%.6f')
    df_data_fmax.to_csv("%s/%s"%(RESULT_DIR, csv_file_name_fmax), mode="a", header=False, float_format='%.6f')
    df_data_fmax_component.to_csv("%s/%s" %(RESULT_DIR, csv_file_name_fmax_component), mode="a", header=False, float_format='%.6f')


def getSPEneryForcesFromFiles(idx):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx % 2)
    model_schnet = load_model("%s/schnetpack/runTraining/%s/best_model" %(BASE_DIR, MODEL_DIR),
                              map_location=device)

    calc_schnet = SpkCalculator(model_schnet, device=device,
                                energy=Properties.energy,
                                forces=Properties.forces,
                                #  collect_triples=True,
                                environment_provider=AseEnvironmentProvider(cutoff=6.0)
                               )

    file_name = file_names[idx]
    file_path = os.path.join(filesDIR, file_name)
    mol = read(file_path)
    n_atoms = mol.get_number_of_atoms()


    mol.set_calculator(calc_schnet)
    schnet_energy = mol.get_potential_energy()
    schnet_fmax = (mol.get_forces()**2).sum(1).max()**0.5  # Maximum atomic force (fom ASE).


    file_base = file_name.split(".")[0]
    energy_values = [file_base,
                     schnet_energy,
                     schnet_energy/n_atoms,
                    ]

    fmax_values = [file_base,
                     schnet_fmax,
                    ]


    for column_name, value in zip(column_names_energy, energy_values):
        df_data_energy[column_name] = [value]

    for column_name, value in zip(column_names_fmax, fmax_values):
        df_data_fmax[column_name] = [value]

    df_data_energy.to_csv("%s/%s" %(RESULT_DIR, csv_file_name_energy), mode="a", header=False, float_format='%.6f')
    df_data_fmax.to_csv("%s/%s"%(RESULT_DIR, csv_file_name_fmax), mode="a", header=False, float_format='%.6f')


def run_multiprocessing(func, argument_list, num_processes):

    pool = Pool(processes=num_processes)

    result_list_tqdm = []

    # implementation of  multiprocessor in tqdm. Ref.https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
    for result in tqdm.tqdm(pool.imap_unordered(func=func, iterable=argument_list), total=len(argument_list)):
        result_list_tqdm.append(result)

    #  return result_list_tqdm


def main(n_procs):
    if mode == "train" or mode == "test":
        idxs = get_trainValTestData_idx(mode)
        print("Nuber of %s data points: %d" %(mode, len(idxs)))
        #  result_list_tqdm = []
        run_multiprocessing(func=getSPEneryForces,
                                           argument_list=idxs,
                                           num_processes=n_procs)
    elif mode == "fromDB":
        n_data = len(data)
        idxs = range(n_data)
        print("Nuber of %s data points: %d" %(mode, len(idxs)))
        #  result_list_tqdm = []
        run_multiprocessing(func=getSPEneryForces,
                                           argument_list=idxs,
                                           num_processes=n_procs)
    elif mode == "fromFiles":
        idxs = range(len(file_names))
        print("Nuber of %s data points: %d" %(mode, len(idxs)))
        #  result_list_tqdm = []
        run_multiprocessing(func=getSPEneryForcesFromFiles,
                                           argument_list=idxs,
                                           num_processes=n_procs)
    else:
        print("Error: Invalid calculation type")
main(8)


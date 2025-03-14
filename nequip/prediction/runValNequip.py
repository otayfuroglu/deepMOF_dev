#

import torch
from nequip.ase import NequIPCalculator
from ase.io import read, write
import tqdm
import argparse


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

    #  return forces[fmax_component_idx[0], fmax_component_idx[1]].item()
    return forces[fmax_component_idx[0]].item()


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-extxyz_path", type=str, required=True, help="")
parser.add_argument("-model_path", type=str, required=True, help="")
args = parser.parse_args()

device = "cuda"

#  model2_path = f"/truba_scratch/otayfuroglu/deepMOF_dev/nequip/works/mof74/runTrain/results/MgF1_nnp2/{version}/MgF1_{version}_nnp2.pth"

extxyz_path = args.extxyz_path
model_path = args.model_path
#  file_base = extxyz_path.split("/")[-1].split(".")[0]
file_base = model_path.split("/")[-1].split(".")[0]

calc = NequIPCalculator.from_deployed_model(
    model_path=model_path,
    device=device,
)

atoms_list = read(extxyz_path, index=":")

fl_enegies = open(f"{file_base}_qm_model_energeis.csv", "w")
fl_forces = open(f"{file_base}_qm_model_forces.csv", "w")
fl_fmax_comp = open(f"{file_base}_qm_model_fmax_comp.csv", "w")
fl_enegies.write(f"index,e_QM,e_Model,e_diff\n")
fl_forces.write(f"index,forces_QM,forces_Model,forces_diff\n")
fl_fmax_comp.write(f"index,fmax_comp_QM,fmax_comp_Model,fmax_comp_diff\n")

#  while n_sample <= 250:
for i in tqdm.trange(0, len(atoms_list), 1):
    #  for i in range(0, len(atoms_list), 1):
    atoms = atoms_list[i]
    #  try:
    label = atoms.info["label"]
    #  except:
    #      label = f"structure_{i}"

    qm_energy = atoms.get_potential_energy() / len(atoms)
    qm_forces = atoms.get_forces().flatten()

    qm_fmax_component_idx = get_fmax_idx(qm_forces)
    qm_fmax_components = get_fmax_componentFrom_idx(qm_forces,
                                                      qm_fmax_component_idx)


    atoms.calc = calc
    model_energy = atoms.get_potential_energy() / len(atoms)
    model_forces = atoms.get_forces().flatten()
    model_fmax_components =  get_fmax_componentFrom_idx(model_forces,
                                                        qm_fmax_component_idx)


    diff_energies = abs(model_energy - qm_energy)
    diff_forces = abs(qm_forces - model_forces)
    diff_fmax_components = abs(qm_fmax_components - model_fmax_components)

    fl_enegies.write(f"{label},{qm_energy},{model_energy},{diff_energies}\n")

    for qm_force, model_force, diff_force in zip(qm_forces, model_forces, diff_forces):
        fl_forces.write(f"{label},{qm_force},{model_force},{diff_force}\n")

    fl_fmax_comp.write(f"{label},{qm_fmax_components},{model_fmax_components},{diff_fmax_components}\n")
    fl_enegies.flush()
    fl_fmax_comp.flush()
    fl_fmax_comp.flush()



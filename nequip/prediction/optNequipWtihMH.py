import numpy as np
from ase.io import read, write
#  from asebazant.bazant_calc import BazantCalculator
from nequip.ase import NequIPCalculator
from ase import units
from ase import Atoms
import minimahopping.opt.optim as opt
import argparse, os
from tqdm import tqdm
import pandas as pd



def lammps2AseAtoms(lammps_atoms, atom_type_symbol_pair):
    symbols = [atom_type_symbol_pair[key] for key in lammps_atoms.get_atomic_numbers()]
    return Atoms(symbols=symbols, positions=lammps_atoms.positions, cell=lammps_atoms.cell)


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-trj_path", type=str, required=True, help="..")
parser.add_argument("-model_path", type=str, required=True, help="")
parser.add_argument("-interval", type=int, required=False, default=1, help="..")
args = parser.parse_args()

device = "cuda"

#  model2_path = f"/truba_scratch/otayfuroglu/deepMOF_dev/nequip/works/mof74/runTrain/results/MgF1_nnp2/{version}/MgF1_{version}_nnp2.pth"

model_path = args.model_path
args = parser.parse_args()
trj_path = args.trj_path

idxs = slice(0, -1, args.interval)
if trj_path.endswith("lammpstrj"):
    atom_type_symbol_pair = {1:"Al", 2:"Li", 3:"H"}
    atoms_list = read(trj_path, format="lammps-dump-text", index=idxs, parallel=True)
elif trj_path.endswith("extxyz"):
    atoms_list = read(trj_path, index=idxs)


#  file_base = extxyz_path.split("/")[-1].split(".")[0]
file_base = trj_path.split("/")[-1].split(".")[0]

calc = NequIPCalculator.from_deployed_model(
    model_path=model_path, device=device,)

#  fl_enegies = open(f"{file_base}_nequip_mh_opt_energeis.csv", "w")
#  fl_enegies.write(f"index,e_Model\n")

calculated_structures_path = f"{file_base}_calculated.csv"
fl_calculated_structures = open(calculated_structures_path, "a")
if os.stat(calculated_structures_path).st_size == 0:
    fl_calculated_structures.write("StrucIdx\n")
fl_calculated_structures.close()

#  while n_sample <= 250:
for i, atoms in enumerate(tqdm(atoms_list)):
    struc_idx = f"struc_{i}"
    calculated_structures = pd.read_csv(calculated_structures_path)["StrucIdx"].to_list()
    if struc_idx in calculated_structures:
        print(f"{struc_idx} has already calculated")
        continue
    else:
        fl_calculated_structures = open(calculated_structures_path, "a")
        fl_calculated_structures.write(f"{struc_idx}\n")
        #  fl_calculated_structures.flush()
        fl_calculated_structures.close()

    #  for i in range(0, len(atoms_list), 1):
    if trj_path.endswith("lammpstrj"):
        atoms = lammps2AseAtoms(atoms, atom_type_symbol_pair)

    atoms.pbc = True

    fmax = 1e-3  # convergence criterion for geometry optimization
    outpath = './' # output path of debug files for gemetry optimization
    initial_step_size = 1e-2 # initial step size
    nhist_max = 10 # maximal history of sqnm
    lattice_weight = 2 # lattice weight for geometry optimization
    alpha_min = 1e-3 # lowest possible step size
    eps_subsp = 1e-3 # subspace stepsize
    verbose_output = False # verbosity of the output

    # perform the geometry optimization
    positions, lattice, noise, opt_trajectory, number_of_opt_steps, epot_max_geopt = opt.optimization(atoms=atoms,
                                                                calculator=calc,
                                                                max_force_threshold=fmax,
                                                                outpath=outpath,
                                                                initial_step_size=initial_step_size,
                                                                nhist_max=nhist_max,
                                                                lattice_weight=lattice_weight,
                                                                alpha_min=alpha_min,
                                                                eps_subsp=eps_subsp,
                                                                verbose=verbose_output)

    # set optimized postions and lattice
    atoms.set_positions(positions)
    atoms.set_cell(lattice)

    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.info["label"] = struc_idx
    write(f"nequip_mh_opt_{file_base}.extxyz", atoms, append=True)


    #  diff_forces = abs(qm_forces - model_forces)

    #  fl_enegies.write(f"{struc_idx},{model_energy}\n")
    #  fl_enegies.flush()



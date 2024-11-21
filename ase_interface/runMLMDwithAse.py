#
from calculationsWithAse import AseCalculations
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import Trajectory
from ase.io.xyz import read_xyz,  write_xyz
from ase.io import read, write
from ase.optimize import BFGS, LBFGS, GPMin, QuasiNewton

import time
import numpy as np
import os, shutil
import argparse
#  import tqdm



def getBoolStr(string):
    string = string.lower()
    if "true" in string or "yes" in string:
        return True
    elif "false" in string or "no" in string:
        return False
    else:
        print("%s is bad input!!! Must be Yes/No or True/False" %string)
        sys.exit(1)


def getWorksDir(calc_name):

    WORKS_DIR = calc_name
    if not os.path.exists(WORKS_DIR):
        os.mkdir(WORKS_DIR)

    return WORKS_DIR


def path2BaseName(path):
    return mol_path.split('/')[-1].split('.')[0]


def checkCalcFiles(mol_name, calculated_names):

    for calculated_name in calculated_names:
        if mol_name in calculated_name:
            return False
    return True

def run(atoms, name, calc_type, temp, replica, fix_indices=[]):
    #calculation.load_molecule_fromFile(mol_path)
    CW_DIR = os.getcwd()
    # main directory for caculation runOpt
    #  if not os.path.exists("ase_worksdir"):
        #  os.mkdir("ase_worksdir")

    WORKS_DIR = getWorksDir(f"{RESULT_DIR}/{name}")
    calculation = AseCalculations(WORKS_DIR)
    calculation.setCalcName(name)
    calculation.load_molecule_fromAseatoms(atoms)
    if pbc:
        calculation.molecule.pbc = True
        if replica >= 2:
            P = [[0, 0, -replica], [0, -replica, 0], [-replica, 0, 0]]
            calculation.makeSupercell(P)
    else:
        calculation.molecule.pbc = False
    os.chdir(WORKS_DIR)

    if calc_type.lower() in ["schnetpack", "ani", "nequip"]:
        import torch
        # in case multiprocesses, global device variable rise CUDA spawn error.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpus = torch.cuda.device_count()
        print("Number of cuda devices --> %s" % n_gpus,)

    if calc_type == "schnetpack":
        from schnetpack.environment import AseEnvironmentProvider
        from schnetpack.utils import load_model
        import schnetpack

        #  model_path = os.path.join(args.MODEL_DIR, "best_model")
        model_schnet = load_model(model_path, map_location=device)
        if "stress" in properties:
            print("Stress calculations are active")
            schnetpack.utils.activate_stress_computation(model_schnet)

        calculation.setSchnetCalcultor(
            model_schnet,
            properties,
            environment_provider=AseEnvironmentProvider(cutoff=5.5),
            device=device,
        )

    elif calc_type == "ani":
        calculation.setAniCalculator(model_type="ani2x", device=device, dispCorrection=None)

    elif calc_type == "nequip":
        calculation.setNequipCalculator(model_path, device)

    elif calc_type.lower() == "n2p2":
        calculation.setN2P2Calculator(
            model_dir=model_path,
            energy_units="eV",
            length_units="Angstrom",
            best_epoch=78)


    # to freeze atoms using constraint
    if len(fix_indices) > 0:
        calculation.setConstraint(fix_indices)

    temperature_K = None
    if md_type == "npt":
        temperature_K = temp

    calculation.init_md(
      name=name,
      time_step=0.5,
      temp_init=temp,
      # temp_bath should be None for NVE and NPT
      temp_bath=temp,
      # temperature_K for NPT
      temperature_K=temperature_K,
      interval=10,
    )

    if opt:
        calculation.optimize(fmax=0.005)
    calculation.run_md(nsteps)

    #  setting strain for pressure deformation simultaions

    #  lattice direction a
    #  abc = calculation.molecule.cell.lengths()
    #  a = abc[0]

    #  for i in range(149):
    #      print("\nStep %s\n" %i)
    #      a -= 0.003 * a
    #      abc[0] = a
    #      calculation.molecule.set_cell(abc)
    #      calculation.run_md(5000)


#  def p_run(idxs):
#      os.environ["CUDA_VISIBLE_DEVICES"] = str(idxs % 2)
#
#      #  db_path = os.path.join(DB_DIR, "nonEquGeometriesEnergyForcesWithORCAFromMD.db")
#      #data = AtomsData(path_to_db)
#      #db_atoms = data.get_atoms(0)
#
#      if n_file > 1:
#          file_name = file_names[idxs]
#          temp = temp_list[0]
#      else:
#          file_name = file_names[0]
#          temp = temp_list[idxs]
#
#      mol_path = os.path.join(MOL_DIR, file_name)
#      run(file_name, mol_path, calc_type, temp, replica)


if __name__ == "__main__":
    #  from multiprocessing import Pool

    parser = argparse.ArgumentParser(description="Give something ...")
    parser.add_argument("-calc_type", type=str, required=True, help="..")
    parser.add_argument("-md_type", type=str, required=True, help="..")
    parser.add_argument("-temp", type=int, required=True, help="..")
    parser.add_argument("-replica", type=int, required=False, default=1, help="..")
    parser.add_argument("-model_path", type=str, required=True, help="..")
    parser.add_argument("-mol_path", type=str, required=True, help="..")
    parser.add_argument("-pbc", type=str, required=True, help="..")
    parser.add_argument("-opt", type=str, required=True, help="..")
    parser.add_argument("-nsteps", type=int, required=True, help="..")
    parser.add_argument("-RESULT_DIR", type=str, required=True, help="..")
    args = parser.parse_args()

    calc_type = args.calc_type
    md_type = args.md_type
    temp = args.temp
    replica = args.replica
    model_path = args.model_path
    mol_path = args.mol_path
    pbc = getBoolStr(args.pbc)
    opt = getBoolStr(args.opt)
    nsteps = args.nsteps
    RESULT_DIR = args.RESULT_DIR
    properties = ["energy", "forces", "stress"]  # properties used for training



    #struc_dir = "ITER2_MOBLEY"
    #calculated_names = os.listdir(RESULT_DIR)
    #print(len(calculated))
    calculated_names = [_dir for _dir in os.listdir(RESULT_DIR)  if os.path.exists(f"{RESULT_DIR}/{_dir}/{_dir}.traj")]
    #print(len(calculated))
    atoms_list=read(mol_path, index=":")
    fl=open("problematic_files.txt", "w")
    for atoms in atoms_list:
        #calc = True

        try:
            label = atoms.info['label']
        except:
            label= "framework"
        name = f"{label}_{calc_type}_{temp}K_{md_type}"
        calc = checkCalcFiles(name, calculated_names)
        if calc:
            #  try:
            #  fix_indices = [atom.index for atom in atoms if atom.symbol == "H"]
            fix_indices = []
            run(atoms, name, calc_type, temp, replica, fix_indices)
           #   except:
           #       print(mol_name, file=fl)


    #  temp_list = [100, 150]
    #  file_names = [file_name for file_name in os.listdir(MOL_DIR) if "." in file_name]
    #  n_file = len(file_names)
    #  if n_file > 1:
    #      idxs = range(n_file)
    #      nprocs = n_file
    #  else:
    #      idxs = range(len(temp_list))
    #      nprocs = len(temp_list)
    #  with Pool(nprocs) as pool:
    #     pool.map(p_run, idxs)

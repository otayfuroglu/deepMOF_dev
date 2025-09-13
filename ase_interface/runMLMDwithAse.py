#
from calculationsWithAse import AseCalculations
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import Trajectory
from ase.io.xyz import read_xyz,  write_xyz
from ase.io import read, write
from ase.optimize import BFGS, LBFGS, GPMin, QuasiNewton

from itertools import combinations
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


def extract_ads(system, ads_atoms, n_frame):
    ref_symbols =ads_atoms.get_chemical_symbols()
    extracted_ads_atoms_list = []
    ads_indices = np.array(range(len(ads_atoms))) + n_frame
    i = ads_indices[-1]

    while ads_indices[-1] < len(system) :
        symbols = [system[i].symbol for i in ads_indices]
        if symbols == ref_symbols:
            #  quit()
            extracted_ads_atoms = system[list(ads_indices)]
            extracted_ads_atoms_list.append(extracted_ads_atoms)
        ads_indices += 1
    return extracted_ads_atoms_list


def extract_ads_indices(system, ads_atoms, n_frame):
    ref_symbols =ads_atoms.get_chemical_symbols()
    ads_indices = np.array(range(len(ads_atoms))) + n_frame
    i = ads_indices[-1]

    extracted_ads_indices = []
    while ads_indices[-1] < len(system) :
        symbols = [system[i].symbol for i in ads_indices]
        if symbols == ref_symbols:
            #  extracted_ads_indices += map(tuple, [ads_indices.tolist()])
            extracted_ads_indices += [ads_indices.tolist()]
        ads_indices += 1
    return  extracted_ads_indices


def run(atoms, name, calc_type, temp, replica):

    #calculation.load_molecule_fromFile(mol_path)
    CW_DIR = os.getcwd()

    # main directory for caculation runOpt
    #  if not os.path.exists("ase_worksdir"):
        #  os.mkdir("ase_worksdir")

    calculation = AseCalculations(WORKS_DIR)
    calculation.setCalcName(name)

    calculation.load_molecule_fromAseatoms(atoms)
    if pbc:
        calculation.molecule.pbc = True
        if replica > 0:
            P = [[0, 0, -replica], [0, -replica, 0], [-replica, 0, 0]]
            calculation.makeSupercell(P)
    else:
        calculation.molecule.pbc = False

    # NOTE  rigid_triatomic works but rigid_triatomic with fixbonds and only fixbonds don't work
    # rigid ads
    # rigid CO2
    #  n_frame = 81
    #  ads_atoms = read(f'{RESULT_DIR}/CO2.xyz') # CO2 molecule
    #  ads_indices_list = extract_ads_indices(atoms, ads_atoms, n_frame)
    #  # for min val in middle
    #  ads_indices_list = [(item[1], item[0], item[2]) for item in ads_indices_list]
    #  calculation.set_rigid_triatomic_atoms(ads_indices_list)
    #
    #  # rigid CH4
    #  ads_atoms = read(f'{RESULT_DIR}/CH4.xyz') # CO2 molecule
    #  ads_indices_list = extract_ads_indices(atoms, ads_atoms, n_frame)
    #  calculation.set_rigid_fixbonds_atoms(ads_indices_list)
    # NOTE

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

    calculation.init_md(
        name=name,
        md_type=md_type,
        time_step=0.5,
        temp_init=temp,
        temp_bath=temp,
        temperature_K=temp,
        pressure=1, #bar
        friction=0.01,          # Friction coefficient for NVT
        ttime=25,               # Thermostat coupling time for NPT
        pfactor=0.6,             # Barostat coupling factor for NPT
        taut=100,               #for NPTBerendsen
        taup=1000,               #for NPTBerendsen
        compressibility=1e-6,   #for NPTBerendsen NPTBerendsen
        reset=False,
        interval=10,
    )

    if opt:
        calculation.optimize(fmax=0.1, steps=200)
    calculation.run_md(nsteps)



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


    #calculated_names = [_dir for _dir in os.listdir(RESULT_DIR)  if os.path.exists(f"{RESULT_DIR}/{_dir}/{_dir}.traj")]
    # calculated_names = [_dir for _dir in os.listdir(RESULT_DIR)]
    # print("\n")
    # print("Nuber of Calculated Files :", len(calculated_names))
    # print("\n")
    #print(len(calculated))
    atoms_list=read(mol_path, index=":")

    fl=open("problematic_files.txt", "w")

    for atoms in atoms_list:
        #calc = True
        atoms.center(vacuum=0.5)

        mol_name = atoms.info['label'].split(".")[0]
        name = f"{mol_name}_{calc_type}_{temp}K_{md_type}"

        #calc = checkCalcFiles(mol_name, calculated_names)
        if not os.path.exists(f"{RESULT_DIR}/{name}"):
            WORKS_DIR = getWorksDir(f"{RESULT_DIR}/{name}")
            print(WORKS_DIR)
            name = f"{mol_name}_{calc_type}_{temp}K_{md_type}"
            #  try:
            run(atoms, name, calc_type, temp, replica)
            #  except:
                #  print(mol_name, file=fl)


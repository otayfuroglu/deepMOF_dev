#
from multiprocessing import Pool
from ase.io import read
import numpy as np
import pandas as pd
import sys, os

import argparse


def calcFramMass(atoms, **atom_mass):
    return sum([len([idx for idx in atoms.get_atomic_numbers()
                     if idx == int(i)]) * atom_mass[i]
                 for i in atom_mass.keys()])


def calcGasMass(atoms, gas_mass, gas_id):
    return sum([len([idx for idx in atoms.get_atomic_numbers()
                     if idx == gas_id]) * gas_mass])


def calcGasNum(atoms, gas_id):
    return len([idx for idx in atoms.get_atomic_numbers()
                     if idx == gas_id])


def create_pd_row(label_val):
    df = pd.DataFrame()
    for label, val in label_val.items():
        df[label] = [val]
    return df


def run(traj_file):
    print("Processing", traj_file)
    # get pressure from file name
    press =int(traj_file.lower()[traj_file.lower().index(gas_type.lower()):traj_file.lower().index("bar")]
               .replace("_", "").replace(gas_type.lower(), ""))

    traj = read("%s/%s" %(traj_dir, traj_file), format="lammps-dump-text", index=":")
    f_start = args.skip
    f_end = len(traj)
    n_frame = f_end - f_start
    fram_mass = calcFramMass(traj[0], **atom_mass)
    gas_loads = np.zeros(n_frame)
    for i, atoms in enumerate(traj[f_start:f_end]):
        gas_loads[i] = calcGasNum(atoms, gas_id)
    avg_gas_loads = gas_loads.mean()
    avg_gas_masses = avg_gas_loads * gas_mass

    print("Pressure: ", press)
    print(f"Number of loaded gas molecule (avg.): {avg_gas_loads}")
    print(f"Weight of loaded gas molecule (avg.): {avg_gas_masses}")
    print(f"wt%: {(avg_gas_masses / fram_mass) * 100}")

    #return as a df
    return create_pd_row({"Pressure": press, "AvgGasLoads": avg_gas_loads,
                              "AvgGasMasses": avg_gas_masses,
                              "AvgGasMasses(wt%)": (avg_gas_masses / fram_mass) * 100})

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Give something ...")
    parser.add_argument("-traj_dir", type=str, required=True)
    parser.add_argument("-gasid", type=int, required=True)
    parser.add_argument("-gastype", type=str, required=True)
    parser.add_argument("-skip", type=int, required=True)
    args = parser.parse_args()

    # Masses
    atom_mass_nnp = {
        "1": 65.3800,
        "2": 15.9994,
        "3": 12.0107,
        "4": 1.00794,
    }

    atom_mass_classic = {
        "1": 65.3800,
        "2": 15.9994,
        "3": 15.9994,
        "4": 12.0107,
        "5": 1.00794,
    }

    gas_id = args.gasid
    gas_type = args.gastype
    traj_dir = args.traj_dir

    if gas_type.lower() == "h2":
        gas_mass = 2.01588
    elif gas_type.lower() == "ch4":
        gas_mass = 16.04246
    else:
        print("Erorr! Please enter gas type as H2 or CH4")
        sys.exit(1)

    atom_mass = atom_mass_nnp
    if gas_id == 6:
        atom_mass = atom_mass_classic

    #  trajfile = "/truba_scratch/yzorlu/deepMOF_dev/n2p2/works/runMD/flexAdorpsionH2onIRMOF1inNPT/IRMOF1_H2_100Bar_77.0K.lammpstrj"
    #  trajfile = "/truba_scratch/yzorlu/deepMOF_dev/n2p2/works/runMD/classicFlexAdorpsionH2onIRMOF1inNPT/IRMOF1_H2_100Bar_77K.lammpstrj"

    traj_files = [traj_file for traj_file in os.listdir(traj_dir) if ".lammpstrj" in traj_file]
    print("Created process up to number of traj_files")
    with Pool(len(traj_files)) as pool:
        dfs = pool.map(run, traj_files)

    df = pd.concat(dfs).reset_index()
    df.to_csv(f"loading_{gas_type}.csv")


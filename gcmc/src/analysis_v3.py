
from ase.io import read
from matplotlib import pyplot as plt
import numpy as np
import os


# Constants
R = 8.314  # Gas constant in J/(mol·K)
N = 6.022e+23  # molecules per mole
BAR2PASCAL = 1e5
#  M_CO2 = 0.04401  # kg/mol for CO2
#  M_CO2 = 44.01  # g/mol for CO2


def calcRhoBulkIdealGas(pressure, temperature):
    """

    Parameters:
    - pressure (float): Pressure (in Pa).
    - temperature (float): Temperature (in K).
    - mol_weight (float): Molar mass of the gas (in kg/mol).

    Returns:
    - rho_bulk_gas (float): (in g/m^3).
    """
    return pressure / (R * temperature)


def getAvgExcess(abs_avg_nAds, void_volume, rho_bulk_gas):

    # Calculate number of molecules of CO2
    moles_co2 = rho_bulk_gas * void_volume
    n_molecules_co2 = moles_co2 * N
    print("# excess CO2", n_molecules_co2)

    avg_nAdsEx = abs_avg_nAds - n_molecules_co2
    return avg_nAdsEx


def getUptakeTraj(extxyz_path, plot=False):
    fl_base = extxyz_path.split("/")[-1].replace(".extxyz", "")
    atoms_list = read(extxyz_path, index=":")
    atoms_frame = atoms_list[0]
    n_atoms_frame = len(atoms_frame)
    nAds_list = [(len(atoms)-n_atoms_frame)/3 for atoms in atoms_list]

    avg_nAds = np.array(nAds_list[int(len(nAds_list)/1.25):]).mean()
    #print(avg_nAds/sum(atoms_frame.get_masses())* 1000, " mmol/g")

    if plot:
        plt.plot(np.array(range(len(nAds_list)))*25, nAds_list)
        plt.xlabel(r"Steps")
        plt.ylabel(r"Number of Molecules")
        plt.savefig(f"{fl_base}.png")
        plt.clf()
        #  plt.show()
    uptake = (avg_nAds +1)/sum(atoms_frame.get_masses())* 1000 # in mmol/g
    return uptake


def getUptakeStatus(csv_path, plot=False):
    import pandas as pd

    fl_base = results_dir
    df = pd.read_csv(csv_path)
    succ_insert = df[" succ_insertion"].tolist()
    succ_del = df["succ_deletion"].tolist()
    nAds_list = [n_insert-n_del for n_insert, n_del in zip(succ_insert,succ_del)]

    avg_nAds = np.array(nAds_list[int(len(nAds_list)/1.25):]).mean()
    #print(avg_nAds/sum(atoms_frame.get_masses())* 1000, " mmol/g")

    if plot:
        plt.plot(np.array(range(len(nAds_list)))*25, nAds_list)
        plt.xlabel(r"Steps")
        plt.ylabel(r"Number of Molecules")
        plt.savefig(f"{fl_base}.png")
        plt.clf()
        #  plt.show()
    uptake = (avg_nAds)/sum(atoms_frame.get_masses())* 1000 # in mmol/g
    return uptake

def getUptakeNpy(npy_path, plot=False):

    fl_base = results_dir
    nAds_list = np.load(npy_path).tolist()

    abs_avg_nAds = np.array(nAds_list[int(len(nAds_list)/2):]).mean()
    excess_avg_nAds = getAvgExcess(abs_avg_nAds, void_volume, rho_bulk_gas)
    #  avg_nAds = np.array(nAds_list).mean()
    #print(avg_nAds/sum(atoms_frame.get_masses())* 1000, " mmol/g")

    if plot:
        plt.plot(np.array(range(len(nAds_list))), nAds_list)
        plt.xlabel(r"Steps")
        plt.ylabel(r"Number of Molecules")
        plt.savefig(f"{fl_base}.png")
        plt.clf()
        #  plt.show()
    abs_uptake = (abs_avg_nAds)/sum(atoms_frame.get_masses())* 1000 # in mmol/g
    excess_uptake = (excess_avg_nAds)/sum(atoms_frame.get_masses())* 1000 # in mmol/g
    #  print(uptake_abs)
    #  uptake_excess = getExcessAdsorption(uptake_abs, V_a, pressure, temperature, M)
    #  print(uptake_excess)
    return abs_uptake, excess_uptake

#  atoms_list = read("./trajectory_1.0bar.extxyz", index=":")
#  extxyz_path = "./trajectory_0.1bar.extxyz"
results_dir_list = [it for it in os.listdir("./") if os.path.isdir(it) and "results" in it]

atoms_frame = read("./frame0.extxyz")

#  mol_weight_co2 = 0.04401 # M_co2 in kg/mol
void_volume = 20* 2.41603e-27 # in m^3


#  atoms_frame = read("./MgMOF74_clean_frame0.extxyz")
fl = open("uptakes.csv", "w")
print("FileName,Pressure(Bar),AbsUptake(mmol/g),ExcesUptake(mmol/g)", file=fl)
for results_dir in results_dir_list:
    pressure = float(results_dir.split("bar")[0].split("_")[-1]) # in bar
    temperature = float(results_dir.split("K")[0].split("_")[-1])
    print(pressure, temperature)
    #  extxyz_path = f"{results_dir}/trajectory_{pressure}bar.extxyz"
    #  csv_path = f"{results_dir}/status.csv"
    npy_path = f"{results_dir}/uptake_{pressure}bar.npy"
    #  uptake = getUptakeTraj(extxyz_path, plot=True)
    pressure *= BAR2PASCAL # in pascal
    rho_bulk_gas = calcRhoBulkIdealGas(pressure, temperature)
    abs_uptake, excess_uptake = getUptakeNpy(npy_path, plot=True)
    print(f"{results_dir},{pressure},{abs_uptake},{excess_uptake}", file=fl)


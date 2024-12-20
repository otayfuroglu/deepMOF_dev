
from ase.io import read
from matplotlib import pyplot as plt
import numpy as np
import os


def getUptake(extxyz_path, plot=False):
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
    uptake = (avg_nAds)/sum(atoms_frame.get_masses())* 1000 # in mmol/g
    return uptake


#  atoms_list = read("./trajectory_1.0bar.extxyz", index=":")
#  extxyz_path = "./trajectory_0.1bar.extxyz"
results_dir_list = [it for it in os.listdir("./") if "results" in it]

fl = open("uptakes.csv", "w")
print("Pressure(Bar),Uptake(mmol/g)", file=fl)
for results_dir in results_dir_list:
    pressure = results_dir.split("bar")[0].split("_")[-1]
    print(pressure)
    extxyz_path = f"{results_dir}/trajectory_{pressure}bar.extxyz"
    uptake = getUptake(extxyz_path, plot=True)
    print(f"{pressure},{uptake}", file=fl)


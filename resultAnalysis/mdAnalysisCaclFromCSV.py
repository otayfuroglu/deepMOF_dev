#! /truba/home/yzorlu/miniconda3/bin/python -u

import pandas as pd
import numpy as np


def calcLengths(volumes):
    return np.cbrt(volumes.mean())


def calcMeanLnVolumes(volumes):
    return np.log(volumes).mean()


def calcVEC():

    mean_ln_volumes = []
    mean_length = []
    job_names = ["%s_%dK_%s_ani" %(file_base, temp, md_type) for temp in temp_list]
    for job_name in job_names:
        #  print(job_name)
        csv_file_path = "%s/%s/timeEnergyTempVolume_%s.csv" % (MD_DIR, job_name, job_name)

        df = pd.read_csv(csv_file_path, skiprows=range(1, initial_skip)) # read colum names and skip

        # set data range from skip initial to end of row according to desired total lenth of data
        df = df[:end_row_idx]
        volumes = df["Volume"].to_numpy()
        mean_ln_volumes.append(calcMeanLnVolumes(volumes))
        mean_length += [calcLengths(volumes)]

    m, b = np.polyfit(temp_list, mean_ln_volumes, 1)

    # save to csv
    df = pd.DataFrame()
    df["Temperature"] = temp_list
    df[""] = mean_length
    df["MeanLength"] = mean_length
    df["LnMeanVolume"] = mean_ln_volumes
    df.to_csv("%s/temperatureLengthLnVolume_%s.csv" %(RESULTS_DIR, file_base))

    return(m, b)


def calcBulkMod(df):
    from ase.units import kJ

    volumes = df["Volume"].to_numpy()
    mean_volumes = volumes.mean()
    sqr_mean_volumes = (volumes ** 2).mean()

    kB = 8.617333e-5 # in eV/K

    B0 = kB * temp * (mean_volumes / (sqr_mean_volumes - mean_volumes ** 2)) # in eV/A^3
    B0 = B0 / kJ * 1.0e24 #  in 'GPa'
    print("Mean a:", calcLengths(volumes))
    print("Bulk Modulus: ", B0)


#  calcVEC()

BASE_DIR = "/truba_scratch/otayfuroglu/deepMOF_dev"
initial_skip = int(3e4)
len_totaldata = 1e5
end_row_idx = int(len_totaldata - initial_skip)
properties = ["energy", "forces"]

sub_jobname = ""
md_type = "md"
MD_DIR = BASE_DIR + "/ani/works/runMD/" + sub_jobname
RESULTS_DIR = BASE_DIR + "/ani/works/runMD/results"


temp_list = np.arange(200, 451, 50)
mof_tec = np.empty((0, 3))

file_bases = ["filled_09000N3"]
for file_base in file_bases:
    print(file_base)

    m, b = calcVEC()
    mof_tec = np.append(mof_tec, np.array([[file_base, m, b]]), axis=0)

# save to csv
df_mof_tec = pd.DataFrame()
df_mof_tec["FileName"] = mof_tec[:, 0]
df_mof_tec["Slope"] = mof_tec[:, 1]
df_mof_tec["Intercept"] = mof_tec[:, 2]
df_mof_tec.to_csv("%s/allMofTEC.csv" %RESULTS_DIR)


file_bases = ["filled_09000N3"]
for file_base in file_bases:
    print(file_base)

    temp = 300
    md_type = "md"
    BASE_DIR = "/truba_scratch/otayfuroglu/deepMOF_dev"
    job_name = "%s_%dK_%s_ani" %(file_base, temp, md_type)
    resut_file_name = "%s_%s.hdf5"  %(file_base, md_type)
    sub_jobname = ""
    MD_DIR = BASE_DIR + "/ani/works/runMD/" + sub_jobname + "/" + job_name
    csv_file_path = "%s/timeEnergyTempVolume_%s.csv" % (MD_DIR, job_name)
    df = pd.read_csv(csv_file_path, skiprows=range(1, initial_skip)) # read colum names and skip

    # set data range from skip initial to end of row according to desired total lenth of data
    df = df[:end_row_idx]
    calcBulkMod(df)

    print("*" * 20)






















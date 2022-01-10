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
    job_names = ["mercury_IRMOF%s_%dK_%s_NPT_111cell_allNNP_without5" % (mof_num, temp, md_type) for temp in temp_list]
    for job_name in job_names:
        #  print(job_name)
        csv_file_path = "%s/timeEnergyTempVolume_%s.csv" % (RESULTS_DIR, job_name)

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
    df.to_csv("%s/temperatureLengthLnVolume_%s.csv" %(RESULTS_DIR, mof_name))

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

BASE_DIR = "/truba_scratch/yzorlu/deepMOF/HDNNP/schnetpack"
initial_skip = int(3e5)
len_totaldata = 1e6
end_row_idx = int(len_totaldata - initial_skip)
properties = ["energy", "forces"]

sub_jobname = "tec_md"
md_type = "md"
RESULTS_DIR = "%s/mdAnalysis/results/%s" % (BASE_DIR, sub_jobname)


temp_list = np.arange(200, 401, 50)
mof_tec = np.empty((0, 3))
for mof_num in [1, 4, 6, 7, 10]:
    mof_name = "IRMOF%s" %mof_num
    print(mof_name)

    m, b = calcVEC()
    mof_tec = np.append(mof_tec, np.array([[mof_name, m, b]]), axis=0)

# save to csv
df_mof_tec = pd.DataFrame()
df_mof_tec["IRMOF%d"%mof_num] = mof_tec[:, 0]
df_mof_tec["Slope"] = mof_tec[:, 1]
df_mof_tec["Intercept"] = mof_tec[:, 2]
df_mof_tec.to_csv("%s/allMofTEC.csv" %RESULTS_DIR)


#  for mof_num in [1, 4, 6, 7, 10]:
#      mof_name = "IRMOF%s" %mof_num
#      print(mof_name)
#
#      temp = 300
#      job_name = "mercury_IRMOF%s_%dK_%s_NPT_111cell_allNNP_without5" % (mof_num, temp, md_type)
#      RESULTS_DIR = "%s/mdAnalysis/results/%s" % (BASE_DIR, sub_jobname)
#      csv_file_path = "%s/timeEnergyTempVolume_%s.csv" % (RESULTS_DIR, job_name)
#      df = pd.read_csv(csv_file_path, skiprows=range(1, initial_skip)) # read colum names and skip
#
#      # set data range from skip initial to end of row according to desired total lenth of data
#      df = df[:end_row_idx]
#      calcBulkMod(df)
#
#      print("*" * 20)






















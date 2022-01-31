#! /truba/home/yzorlu/miniconda3/bin/python

import pandas as pd
import matplotlib.pyplot as plt


def plot_phonon_dos():
    wave_num = df[0]
    dos = df[1]

    #  Plot the energies
    plt.figure()
    plt.plot(wave_num, dos, c=colors[idx])
    plt.ylabel('E [kcal/mol]')
    plt.xlabel('t [fs]')
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s/phonon_dos.png" % RESULTS_DIR, db=600)


for idx, mof_num in enumerate([1, 4, 6, 7, 10]):

    BASE_DIR = "/truba_scratch/yzorlu/deepMOF/HDNNP/schnetpack"
    RESULTS_DIR = BASE_DIR + "/asePlatform/run_worksdir/mercury_IRMOF%s_model_vibration_300K" %mof_num
    data_file_path = "%s/vib-dos.dat" % RESULTS_DIR
    df = pd.read_csv(data_file_path,
                     delim_whitespace=True, comment='#', header=None)
    maker_types = ["v", "8", "s", "o", "x"]
    colors = ["b", "m", "y", "g", "c"]

    plot_phonon_dos()


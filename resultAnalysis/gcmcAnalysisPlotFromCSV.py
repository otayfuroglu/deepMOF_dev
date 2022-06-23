#! /home/omert/miniconda3/bin/python


import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

sns.set_context("paper", rc={"grid.linewidth": 0.8})
sns.set_style("ticks", rc={"grid.linestyle": "--"})

def plot_uptakeStrain(df):
    labels = ["Uptakes (mg/g)", "Uptakes (cm^3 (STP)/cm^3)"]
    strains = df["Strain"]
    uptakes_grav = df[labels[0]]
    uptakes_vol = df[labels[1]]


    #  Plot the energies
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(strains, uptakes_grav, "o", c=colors[0], label=r"${%s}$"%labels[0])
    ax2.plot(strains, uptakes_vol, "P",  c=colors[1], label=r"${%s}$"%labels[1])

    ax1.set_ylabel(r"$CH_4$ Uptake $(mg / g)$")
    ax2.set_ylabel(r"$CH_4$ Uptake ($cm^3$ STP / $cm^3)$")
    ax1.set_xlabel(r"Compressive Strain (%)")

    #  ax1.set_ylim(25, 80)
    #  ax2.set_ylim(350, 550)
    ax2.yaxis.label.set_color(colors[1])
    ax2.tick_params(axis='y', colors=colors[1])
    ax2.spines["right"].set_edgecolor(colors[1])

    ax1.legend(loc="upper center", prop={'size': 8.5}, bbox_to_anchor=(0.41, 0.26),
              fancybox=False, shadow=False, labelcolor=colors[0], frameon=False)
    ax2.legend(loc="upper center", prop={'size': 8.5}, bbox_to_anchor=(0.46, 0.20),
              fancybox=False, shadow=False, labelcolor=colors[1], frameon=False)



    fig.tight_layout()
    plt.savefig("%s/%s.png" % (RESULTS_DIR, job_name), db=600)
    #  plt.show()


def plot_ASA_POAV(df):
    labels = ["Density (g/L)", "POAV Fraction (%)"]
    #  labels = ["ASA (m^2/g)", "POAV (cm^3/g)"]
    strains = df["Strain"]
    uptakes_grav = df[labels[0]]
    uptakes_vol = df[labels[1]]


    #  Plot the energies
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(strains, uptakes_grav, "o", c=colors[0], label=r"%s"%labels[0], alpha=0.99)
    ax2.plot(strains, uptakes_vol, "^",  c=colors[1], label=r"%s"%labels[1], alpha=0.99)

    ax1.set_ylabel(r"%s"%labels[0])
    ax2.set_ylabel(r"%s"%labels[1])
    ax1.set_xlabel(r"Compressive Strain (%)")

    ax2.yaxis.label.set_color(colors[1])
    ax2.tick_params(axis='y', colors=colors[1])
    ax2.spines["right"].set_edgecolor(colors[1])

    ax1.legend(loc="upper center", prop={'size': 8.5}, bbox_to_anchor=(0.45, 0.23),
              fancybox=False, shadow=False, labelcolor=colors[0], frameon=False)
    ax2.legend(loc="upper center", prop={'size': 8.5}, bbox_to_anchor=(0.48, 0.18),
              fancybox=False, shadow=False, labelcolor=colors[1], frameon=False)



    #  fig.legend(loc="upper center", prop={'size': 8.5}, bbox_to_anchor=(0.58, 0.95))
    fig.tight_layout()
    plt.savefig("%s/%s_density_volume_frac.png" % (RESULTS_DIR, job_name), db=600)
    #  plt.savefig("%s/%s.png" % (RESULTS_DIR, job_name), db=600)
    #  plt.show()


for idx, mof_num in enumerate([1, 7]):

    md_type = "md"
    BASE_DIR = "/home/omert/Desktop/deepMOF/deepMOF/HDNNP/schnetpack"
    RESULTS_DIR ="%s/gcmcAnalysis" %BASE_DIR
    maker_types = ["v", "8", "s", "o", "x"]
    colors = ["k",  "b", "midnightblue", "darkred", "firebrick", "b", "r", "dimgray", "orange", "m", "y", "g", "c"]

    # for gas uptake
    #  job_name = "IRMOF%d_methane_uptakes_298K_35Bar_CrytalGen" % mof_num
    #  #  job_name = "IRMOF%d_H2_uptakes_77K_35Bar_CrytalGen" % mof_num
    #  csv_path = "%s/%s.csv" % (RESULTS_DIR, job_name)
    #  df = pd.read_csv(csv_path)
    #  plot_uptakeStrain(df)


    # for ASa POAV
    job_name = "./IRMOF%d_all_asa_poav" % mof_num
    csv_path = "%s/%s.csv" % (RESULTS_DIR, job_name)
    df = pd.read_csv(csv_path)
    plot_ASA_POAV(df)



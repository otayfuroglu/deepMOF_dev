#
#from ase import Atoms
#from ase.io import read
#from ase.db import connect
#
#from schnetpack.utils import load_model
#from schnetpack.datasets import *

import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import make_interp_spline
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

import os, warnings
index_warning = 'Converting sparse IndexedSlices'
warnings.filterwarnings('ignore', index_warning)

#  sns.set()
sns.set_context("paper", rc={"grid.linewidth": 0.8})
sns.set_style("ticks", rc={"grid.linestyle": "--"})
#  plt.style.use('seaborn-ticks')


# path definitions
#model_schnet = load_model("./mof5_model_hdnnp_forces_v4/best_model")
##model_schnet = load_model("./ethanol_model/best_model")
#properties = ["cohesive_E_perAtom"]#, "forces"]  # properties used for training
#

#df = df_data.astype(str).groupby("FileNames").agg(";".join)
#  mof5_f1 = df_data.loc[df_data["FileNames"].str.contains("mof5_f1")]
#  print(mof5_f1.head())

mof_num = 4
RESULT_DIR = "/truba_scratch/yzorlu/deepMOF/HDNNP/schnetpack/results/IRMOF%s" % mof_num
NAME_BASE = "irmofseries"
mof_name = NAME_BASE + str(mof_num)
frag_name = mof_name + "_f6"
idx = 0 # for shape and color

def plot_linear_reg(val_type, prop, calc_type, RESULT_DIR):


    maker_types = ["v", "8", "s", "o", "x"]
    colors = ["b", "m", "y", "g", "c"]

    plt.rcParams["figure.figsize"] = (16, 10)
    fig, ax = plt.subplots() # created subplots which share y axis 

    if prop == "E":
        fig.text(0.5, 0.04, "$E_{DFT}$ / eV", ha="center", va="center", size=14)
        fig.text(0.05, 0.5, "$E_{NNP}$ / eV", ha="center", va="center", rotation=90, size=14)
    if prop == "F" or prop == "FC":
        fig.text(0.5, 0.04, r"$F_{DFT}$ / eV$\AA$$^{-1}$", ha="center", va="center", size=14)
        fig.text(0.05, 0.5, r"$F_{NNP}$ / eV$\AA$$^{-1}$", ha="center", va="center", rotation=90, size=14)


    csv_file_path = "%s/qm_sch_SP%s_%s_%s.csv" %(RESULT_DIR, calc_type, prop, val_type)
    df_mof = pd.read_csv(csv_file_path)

    print("Property %s: %s --> " %(prop, frag_name), len(df_mof))
    #  assert len(df_mof) == 0, "Empty DATA !!!"

    x = df_mof.iloc[:, 2]
    y = df_mof.iloc[:, 3]

    m, b = np.polyfit(x, y, 1) # get regression line paremeres
    linreg = stats.linregress(x, y)
    ax.plot(x, m * x + b, ls="-", color="k", linewidth=0.8) # plot regression line

    ax.text(x.max()-3, y.max(), "$R^2$=%.3f"%linreg.rvalue,
            color="k", ha="right", va="center", size=8)


    ax.scatter(x=x, y=y, c=colors[idx], s=10,
                  marker=maker_types[idx], alpha=0.5,
                  label=frag_name)
    ax.legend()

    ax.grid(which='both', axis='both',  linestyle='--', linewidth=0.8)
    ax.ticklabel_format(useOffset=False)
    if prop == "E":
        #  ax.ticklabel_format(style='scientific', axis='both', scilimits=(0, 0))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

        if val_type == "test":
           ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
           ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        #  ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
        #  ax.xaxis.set_ticks([x.min(), y.max()])

        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

        if val_type == "test":
           ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
           ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
        #  ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
        #  ax.yaxis.set_ticks([x.min(), y.max()])

        #  ax.tick_params(axis="both", labelsize="small", length=6, width=2)
        #  ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))



    #  ax.text(df_mof.iloc[:, 5].max(), 0.03, "mof5_f%s: %d adet" %((i+1),len(df_mof)),
            #  color="k", ha="right", va="center", size=8)

    #  plt.show()
    plt.savefig("%s/%s_lineerRegression%s_%s_%s" %(RESULT_DIR, mof_name, prop, val_type, calc_type), dpi=600)
#  plot_linear_reg("E", df_data)


def plot_error(val_type, prop, calc_type, RESULT_DIR):

    # get single mof data
    csv_file_path = "%s/qm_sch_SP%s_%s_%s.csv" %(RESULT_DIR, calc_type, prop, val_type)
    df_mof = pd.read_csv(csv_file_path)
    print("Property %s: %s --> " %(prop, frag_name), len(df_mof))

    maker_types = ["v", "8", "s", "o", "x"]
    colors = ["b", "m", "y", "g", "c"]

    plt.rcParams["figure.figsize"] = (12, 6)
    fig, ax = plt.subplots()


    # set common x and y axis label
    if prop == "E":
        fig.text(0.5, 0.04, "$E_{DFT}$ / eV $atom^{-1}}$", ha="center", va="center", size=14)
        fig.text(0.05, 0.5, "$E_{NNP}$ - $E_{DFT}$ / eV $atom^{-1}$", ha="center", va="center", rotation=90, size=14)
    if prop == "F" or prop == "FC":
        fig.text(0.5, 0.04, r"$F_{DFT}$ / eV $\AA$$^{-1}$", ha="center", va="center", size=14)
        fig.text(0.05, 0.5, r"$F_{NNP}$ - $F_{DFT}$ / eV $\AA$$^{-1}$", ha="center", va="center", rotation=90, size=14)

    ax.grid(which='both', axis='both',  linestyle='--', linewidth=0.8)
    ax.ticklabel_format(useOffset=False)

    if prop == "E":
        x = df_mof.iloc[:, 5]
        y = df_mof.iloc[:, 7]
        ax.set_ylim(-0.2, 0.2)
        # add horizontal line
        ax.axhline(y=0.005, color="r", linewidth=0.8, linestyle='dashed') #, label='p=[0.005 - 0.005]')
        ax.axhline(y=-0.005, color="r", linewidth=0.8, linestyle='dashed') #', label='p=-0.05')
        #  ax.text(x.max(), 0.03, "mof5_f%s: %d adet" %((i+1),len(df_mof)),
        #          color="k", ha="right", va="center", size=10)

    if prop == "F" or prop == "FC":
        x = df_mof.iloc[:, 2]
        y = df_mof.iloc[:, 4]
        ax.set_ylim(-10, 10)
        # add horizontal line
        ax.axhline(y=0.5, color="r", linewidth=0.8, linestyle='dashed') #, label='p=[0.005 - 0.005]')
        ax.axhline(y=-0.5, color="r", linewidth=0.8, linestyle='dashed') #', label='p=-0.05')
        #  ax.text(x.max(), 3, "mof5_f%s: %d adet" %((i+1),len(df_mof)),
        #          color="k", ha="right", va="center", size=10)
    #  ax.legend()

    ax.scatter(x=x, y=y, c=colors[idx], s=1, marker=maker_types[idx], label="%s: %d adet" %(frag_name, len(df_mof)))
    ax.legend()

    # text label for axhline
    #  ax.text(df_mof.iloc[:, 5].min(), 0.007, "{:.3f}".format(0.005), color="b", ha="right", va="center", size=8)
    #  ax.text(df_mof.iloc[:, 5].min(), -0.009, "{:.3f}".format(-0.005), color="b", ha="right", va="center", size=8)

    #  plt.show()
    plt.savefig("%s/%s_Errors%s_%s_%s" %(RESULT_DIR, mof_name, prop, val_type, calc_type), dpi=600)

def plot_histograms(val_type, prop, calc_type, RESULT_DIR):

    # get single mof data
    csv_file_path = "%s/qm_sch_SP%s_%s_%s.csv" %(RESULT_DIR, calc_type, prop, val_type)
    df_mof = pd.read_csv(csv_file_path)
    print("Property %s: %s --> " %(prop, frag_name), len(df_mof))

    maker_types = ["v", "8", "s", "o", "x"]
    colors = ["b", "m", "y", "g", "c"]

    plt.rcParams["figure.figsize"] = (6, 6)
    fig, ax = plt.subplots()

    # set common x and y axis label
    if prop == "E":
        fig.text(0.5, 0.04, "$E_{NNP}$ - $E_{DFT}$ / eV $atom^{-1}$", ha="center", va="center", size=12)
        fig.text(0.03, 0.5, "Frequency", ha="center", va="center", rotation=90, size=12)
    if prop == "F" or prop == "FC":
        fig.text(0.5, 0.04, r"$F_{NNP}$ - $F_{DFT}$ / eV $\AA$$^{-1}$", ha="center", va="center", size=12)
        fig.text(0.03, 0.5, "Frequency", ha="center", va="center", rotation=90, size=12)

    ax.grid(which='both', axis='both',  linestyle='--', linewidth=0.8)
    ax.ticklabel_format(useOffset=False)

    if prop == "E":
        x = df_mof.iloc[:, 7]
        ax.hist(x=x, bins=20, color = colors[idx],
                       range=(-0.05, 0.05),
                       label=frag_name,
                      )
    if prop == "F" or prop == "FC":
        x = df_mof.iloc[:, 4]
        ax.hist(x=x, bins=20, color = colors[idx],
                       range=(-1, 1),
                       label=frag_name)
    ax.legend()


    #  ax.text(df_mof.iloc[:, 5].max(), 0.03, "mof5_f%s: %d adet" %((i+1),len(df_mof)),
                #  color="k", ha="right", va="center", size=8)

    #  plt.show()
    plt.savefig("%s/%s_Histograms%s_%s_%s" %(RESULT_DIR, mof_name, prop, val_type, calc_type), dpi=600)

def plotTorsionPhenyl(RESULT_DIR):

    plt.rcParams["figure.figsize"] = (16, 10)
    fig, ax = plt.subplots() # created subplots which share y axis 

    csv_file_path = "%s/qm_sch_SP_E_torsion.csv" %(RESULT_DIR)
    df_mof = pd.read_csv(csv_file_path)

    #  assert len(df_mof) == 0, "Empty DATA !!!"



    angles = []
    for i, row in df_mof.iterrows():
        file_name = row["FileNames"]
        angle = file_name.split("_")[-1].replace("deg", "")
        angles.append(angle)

    df_mof["Angles"] = angles

    df_mof_dftE = df_mof[["Angles", "qm_SP_energies"]]
    df_mof_dftE["Angles"] = df_mof_dftE["Angles"].astype(float)
    df_mof_dftE= df_mof_dftE.sort_values(by="Angles")

    df_mof_schnetE = df_mof[["Angles", "schnet_SP_energies"]]
    df_mof_schnetE["Angles"] = df_mof_schnetE["Angles"].astype(float)
    df_mof_schnetE = df_mof_schnetE.sort_values(by="Angles")

    # substruct 0 degree values from all degree values
    df_mof_dftE["qm_SP_energies"] = df_mof_dftE["qm_SP_energies"] - df_mof_dftE.loc[df_mof_dftE["Angles"] == 0]["qm_SP_energies"].item()
    df_mof_schnetE["schnet_SP_energies"] = df_mof_schnetE["schnet_SP_energies"] - df_mof_schnetE.loc[df_mof_schnetE["Angles"] == 0]["schnet_SP_energies"].item()

    # insert value of 180 degree and regarding energy as zero to data
    df_mof_dftE.loc[-1] = [180, 0]
    df_mof_schnetE.loc[-1] = [180, 0]

    maker_types = ["v", "8", "s", "o", "x"]
    colors = ["b", "m", "y", "g", "c"]

    plt.rcParams["figure.figsize"] = (6, 6)
    fig, ax = plt.subplots()

    # set common x and y axis label
    fig.text(0.5, 0.04, r"$\Phi$ / $\circ$ ", ha="center", va="center", size=14)
    fig.text(0.05, 0.5, r"$\Delta$E / eV", ha="center", va="center", rotation=90, size=14)

    ax.grid(which='both', axis='both',  linestyle='--', linewidth=0.8)
    ax.ticklabel_format(useOffset=False)

    #  ax.text(x.max()-3, y.max(), "$R^2$=%.3f"%linreg.rvalue,
    #          color="k", ha="right", va="center", size=8)


    idx = 0
    ax.scatter(x=df_mof_dftE.Angles, y=df_mof_dftE.qm_SP_energies, c=colors[idx], s=15,
                  marker=maker_types[idx], alpha=0.5,
                  label=mof_name)
    idx = 1
    ax.scatter(x=df_mof_schnetE.Angles, y=df_mof_schnetE.schnet_SP_energies, c=colors[idx], s=15,
                  marker=maker_types[idx], alpha=0.5)
    ax.legend()


    plt.savefig("%s/%s_torsionPhenyl.png" %(RESULT_DIR, mof_name), dpi=600)


def plotStrengthCH(RESULT_DIR, val_type):

    plt.rcParams["figure.figsize"] = (16, 10)
    fig, ax = plt.subplots() # created subplots which share y axis 

    csv_file_path = "%s/qm_sch_SP_E_%s_bond.csv" %(RESULT_DIR, val_type)
    df_mof = pd.read_csv(csv_file_path)

    #  assert len(df_mof) == 0, "Empty DATA !!!"



    chBonds = []
    for i, row in df_mof.iterrows():
        file_name = row["FileNames"]
        chBond = float(file_name.split("_")[-1]) / 100
        chBonds.append(chBond)

    df_mof["ChBonds"] = chBonds

    df_mof_dftE = df_mof[["ChBonds", "qm_SP_energies"]]
    df_mof_dftE= df_mof_dftE.sort_values(by="ChBonds")

    df_mof_schnetE = df_mof[["ChBonds", "schnet_SP_energies"]]
    df_mof_schnetE = df_mof_schnetE.sort_values(by="ChBonds")

    # substruct 0 degree values from all degree values
    df_mof_dftE["qm_SP_energies"] = df_mof_dftE["qm_SP_energies"] - df_mof_dftE.loc[df_mof_dftE["ChBonds"] == 1.09]["qm_SP_energies"].item()
    df_mof_schnetE["schnet_SP_energies"] = df_mof_schnetE["schnet_SP_energies"] - df_mof_schnetE.loc[df_mof_schnetE["ChBonds"] == 1.09]["schnet_SP_energies"].item()

    # insert value of 180 degree and regarding energy as zero to data
    #  df_mof_dftE.loc[-1] = [180, 0]
    #  df_mof_schnetE.loc[-1] = [180, 0]

    maker_types = ["v", "8", "s", "o", "x"]
    colors = ["b", "m", "y", "g", "c"]

    plt.rcParams["figure.figsize"] = (6, 6)
    fig, ax = plt.subplots()

    # set common x and y axis label
    fig.text(0.5, 0.04, r"r(C-H)$\AA$", ha="center", va="center", size=14)
    fig.text(0.05, 0.5, r"$\Delta$E / eV", ha="center", va="center", rotation=90, size=14)

    ax.grid(which='both', axis='both',  linestyle='--', linewidth=0.8)
    ax.ticklabel_format(useOffset=False)

    #  ax.text(x.max()-3, y.max(), "$R^2$=%.3f"%linreg.rvalue,
    #          color="k", ha="right", va="center", size=8)

    # for smooth line on dft energies
    xnew = np.linspace(df_mof_dftE.ChBonds.min(), df_mof_dftE.ChBonds.max())
    gfg = make_interp_spline(df_mof_dftE.ChBonds, df_mof_dftE.qm_SP_energies, k=3)
    ynew = gfg(xnew)
    ax.plot(xnew, ynew)

    idx = 3
    ax.scatter(x=df_mof_dftE.ChBonds, y=df_mof_dftE.qm_SP_energies, c=colors[idx], s=20,
                  marker=maker_types[idx], alpha=0.5,
                  label="%s DFT" %mof_name)
    idx = 1
    ax.scatter(x=df_mof_schnetE.ChBonds, y=df_mof_schnetE.schnet_SP_energies, c=colors[idx], s=20,
               marker=maker_types[idx], alpha=0.5,
               label="%s NNP" %mof_name

              )
    ax.legend()


    plt.savefig("%s/%s_strength_%s.png" %(RESULT_DIR, mof_name, val_type), dpi=600)


def plotModelsRMSDiff():

    maker_types = ["v", "8", "s", "o", "x"]
    colors = ["b", "m", "y", "g", "c"]

    plt.rcParams["figure.figsize"] = (8, 5)
    fig, ax = plt.subplots() # created subplots which share y axis 

    # set common x and y axis label
    fig.text(0.5, 0.04, r"Data Sayısı", ha="center", va="center", size=14)
    fig.text(0.05, 0.5, r"$E_{NNP1}$ - $E_{NNP2}$ / eV", ha="center", va="center", rotation=90, size=14)

    ax.grid(which='both', axis='both',  linestyle='--', linewidth=0.8)
    ax.ticklabel_format(useOffset=False)

    csv_file_path = "./IRMOFseries6_models_rms_diff.csv"
    df_mof = pd.read_csv(csv_file_path)

    #  assert len(df_mof) == 0, "Empty DATA !!!"



    lenDatas = []
    for i, row in df_mof.iterrows():
        file_name = row["ModelsVersion"]
        lenData = file_name.split("_")[-2]
        lenDatas.append(float(lenData))

    df_mof["LenData"] = lenDatas

    df_mof_MADiff = df_mof[["LenData", "MADiff"]]
    df_mof_MADiff = df_mof_MADiff.sort_values(by="LenData")
    df_mof_RMSDiff = df_mof[["LenData", "RMSDiff"]]
    df_mof_RMSDiff = df_mof_RMSDiff.sort_values(by="LenData")

    idx = 2
    ax.scatter(x=df_mof_MADiff.LenData, y=df_mof_MADiff.MADiff, c=colors[idx], s=20,
                  marker=maker_types[idx], alpha=0.5,
                  label="MA Belirsizlik")
    idx = 0
    ax.scatter(x=df_mof_RMSDiff.LenData, y=df_mof_RMSDiff.RMSDiff, c=colors[idx], s=20,
               marker=maker_types[idx], alpha=0.5,
               label="RMS Belirsizlik"
              )
    ax.legend()

    plt.savefig("IRMOFseries6_models_rms_diff.png", dpi=600)

#  plotModelsRMSDiff()
plotStrengthCH(RESULT_DIR, val_type="aliphatic_ch")
#  plotTorsionPhenyl(RESULT_DIR)

#  if __name__ == "__main__":
#     for calc_type in [ "", ]: #"Ensemble",]:
#         for val_type in ["train", "test"]:
#             for prop in ["E", "FC"]: # "F"
#
#                 #  column_names = ["FileNames", "qm_SP_energies", "schnet_SP_energies", "Error", "ErrorPerAtom"]
#
#                 plot_linear_reg(val_type, prop, calc_type, RESULT_DIR)
#                 plot_error(val_type, prop, calc_type, RESULT_DIR)
#                 plot_histograms(val_type, prop, calc_type, RESULT_DIR)
#

#! /home/omert/miniconda3/bin/python
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

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
matplotlib.use("Agg")

from sklearn import preprocessing
from scipy.stats import norm, gaussian_kde
#  import mpl_scatter_density # adds projection='scatter_density'

from multiprocessing import Pool

import os, warnings
index_warning = 'Converting sparse IndexedSlices'
warnings.filterwarnings('ignore', index_warning)

#  sns.set()
sns.set_context("paper", rc={"grid.linewidth": 0.8})
sns.set_style("ticks", rc={"grid.linestyle": "--"})
#  plt.style.use('seaborn-ticks')


def normalizeData(data):
    return preprocessing.minmax_scale(data, feature_range=(-100, -0))


def calc_kernel(p_y):
    return gaussian_kde(p_y)(p_y)


def _plot_error_norm(i, fig, axs, df_mof, density=False):

    maker_types = ["v", "8", "s", "o", "x", "4", "+", "*", "<"]
    colors = ["navy", "c", "lightblue", "indigo", "lime", "aqua", "orange", "chocolate", "purple"]

    axs.grid(which='both', axis='both',  linestyle='--', linewidth=0.8)
    axs.ticklabel_format(useOffset=False)

    if prop == "E":
        x = df_mof.iloc[:, 5]
        x = normalizeData(x)
        y = df_mof.iloc[:, 7]
        axs.set_ylim(-0.05, 0.05)
        # add horizontal line
        axs.axhline(y=0.005, color="k", linewidth=1.0, linestyle='dashed') #, label='p=[0.005 - 0.005]')
        axs.axhline(y=-0.005, color="k", linewidth=1.0, linestyle='dashed') #', label='p=-0.05')
        #  axs[i].text(x.max(), 0.03, "mof5_f%s: %d adet" %((i+1),len(df_mof)),
        #              color="k", ha="right", va="center", size=10)

    if prop == "F" or prop == "FC" or prop == "FAll":
        x = df_mof.iloc[:, 2]
        y = df_mof.iloc[:, 4]
        axs.set_ylim(-5, 5)
        if val_type == "train":
            axs.set_xlim(-45, 45)
        elif val_type == "test":
            axs.set_xlim(-45, 45)

        # add horizontal line
        axs.axhline(y=0.5, color="k", linewidth=1.0, linestyle='dashed') #, label='p=[0.005 - 0.005]')
        axs.axhline(y=-0.5, color="k", linewidth=1.0, linestyle='dashed') #', label='p=-0.05')
        #  axs[i].text(x.max(), 3, "mof5_f%s: %d adet" %((i+1),len(df_mof)),
        #              color="k", ha="right", va="center", size=10)
    #  axs[i].legend()

    if density:

        n_proc = 56

        p_y = np.array_split(y, n_proc)
        pool = Pool(processes=n_proc)
        results = pool.map(calc_kernel, p_y)
        c = np.concatenate(results)

        #  c =  gaussian_kde(y)(y)

        # for the number of  train data scaling
        if prop == "FC":
            c = c * 80

        density = axs.scatter(x=x, y=y, c=c, cmap="turbo", s=3, marker="o", edgecolor=None)
        fig.colorbar(density, label="Counts")
    else:
        axs.scatter(x=x, y=y, c=colors[i], s=3, marker=maker_types[i], alpha=0.8, label="F%d" % (i+1))
        axs.legend(prop={'size': 10}, loc='upper left', ncol=n_frags, markerscale=2.0)

    # text label for axhline
    #  axs[i].text(df_mof.iloc[:, 5].min(), 0.007, "{:.3f}".format(0.005), color="b", ha="right", va="center", size=8)
    #  axs[i].text(df_mof.iloc[:, 5].min(), -0.009, "{:.3f}".format(-0.005), color="b", ha="right", va="center", size=8)
    return fig, axs

def plot_error_norm_density(val_type, prop, calc_type, df_data):

    #  if single_mof_idx == 1:
    #      NAME_BASE = "mof"
    #      single_mof_idx = 5
    #  else:
    #      NAME_BASE = "irmofseries"

    plt.rcParams["figure.figsize"] = (16, 4.5)
    plt.rcParams["font.size"] = 14
    #  fig, axs = plt.subplots(n_frags, 1, sharey=True) # created subplots which share y axis
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1) #, projection="scatter_density")

    # set common x and y axis label
    if prop == "E":
        axs.set_xlabel("$E_{DFT}$ / eV $atom^{-1}}$ (Normalized)", size=12)
        axs.set_ylabel("$E_{NNP}$ - $E_{DFT}$ / eV $atom^{-1}$", size=12)
    if prop == "F" or prop == "FC":
        axs.set_xlabel(r"$F_{DFT}$ / eV $\AA$$^{-1}$", size=12)
        axs.set_ylabel(r"$F_{NNP}$ - $F_{DFT}$ / eV $\AA$$^{-1}$", size=12)
    norm_x = []
    y = []
    mof_nums = [4, 6, 7, 10] # mof numbers exclude IRMOF-1
    for i in range(n_frags):
        if i <= 4:
            mof_name = "mof5"
            df_mof = df_data.loc[df_data["FileNames"].str.contains(mof_name)]

            frag_base = "_f" + str(i+1)
            df_mof = df_mof.loc[df_mof["FileNames"].str.contains(frag_base)]

            if prop == "E":
                x = df_mof.iloc[:, 5]
                x = normalizeData(x)
                norm_x += list(x)
                y += list(df_mof.iloc[:, 7])

        else:
            #  for mof_num in IDX_MOFs[1:]:
            mof_num = mof_nums.pop(0)
            mof_name = "irmofseries" + str(mof_num)
            df_mof = df_data.loc[df_data["FileNames"].str.contains(mof_name)]
            #  frag_base = "_f" + str(idx+1)
            frag_base = "_f6"
            df_mof = df_mof.loc[df_mof["FileNames"].str.contains(frag_base)]

            if prop == "E":
                x = df_mof.iloc[:, 5]
                x = normalizeData(x)
                norm_x +=list(x)
                y += list(df_mof.iloc[:, 7])

    if prop == "E":
        df_data.iloc[:, 5] = norm_x
        df_data.iloc[:, 7] = y
    fig, axs = _plot_error_norm(i, fig, axs, df_mof=df_data, density=True)
    #  plt.show()
    plt.savefig("%s/ErrorsCounts_%s_%s_%s" %(RESULT_DIR, prop, val_type, calc_type), dpi=600)
#  plot_error("E", df_data)

def plot_error_norm(val_type, prop, calc_type, df_data):

    #  if single_mof_idx == 1:
    #      NAME_BASE = "mof"
    #      single_mof_idx = 5
    #  else:
    #      NAME_BASE = "irmofseries"

    plt.rcParams["figure.figsize"] = (16, 4.5)
    plt.rcParams["font.size"] = 14
    #  fig, axs = plt.subplots(n_frags, 1, sharey=True) # created subplots which share y axis
    fig, axs = plt.subplots(1, 1)

    # set common x and y axis label
    if prop == "E":
        axs.set_xlabel("$E_{DFT}$ / eV $atom^{-1}}$", size=12)
        axs.set_ylabel("$E_{NNP}$ - $E_{DFT}$ / eV $atom^{-1}$", size=12)
    if prop == "F" or prop == "FC":
        axs.set_xlabel(r"$F_{DFT}$ / eV $\AA$$^{-1}$", size=12)
        axs.set_ylabel(r"$F_{NNP}$ - $F_{DFT}$ / eV $\AA$$^{-1}$", size=12)

    mof_nums = [4, 6, 7, 10] # mof numbers exclude IRMOF-1
    for i in range(n_frags):
        if i <= 4:
            mof_name = "mof5"
            df_mof = df_data.loc[df_data["FileNames"].str.contains(mof_name)]

            frag_base = "_f" + str(i+1)
            df_mof = df_mof.loc[df_mof["FileNames"].str.contains(frag_base)]
            print("Property %s: %s%s --> " %(prop, mof_name, frag_base), len(df_mof))
            fig, axs = _plot_error_norm(i, fig, axs, df_mof)

        else:
            #  for mof_num in IDX_MOFs[1:]:
            mof_num = mof_nums.pop(0)
            mof_name = "irmofseries" + str(mof_num)
            df_mof = df_data.loc[df_data["FileNames"].str.contains(mof_name)]
            #  frag_base = "_f" + str(idx+1)
            frag_base = "_f6"
            df_mof = df_mof.loc[df_mof["FileNames"].str.contains(frag_base)]
            print("Property %s%s: %s --> " %(prop, mof_name, frag_base), len(df_mof))
            #  df_mof = df_mof.loc[df_mof["FileNames"].str.contains("co2") == False] # does not contain.

            fig, axs = _plot_error_norm(i, fig, axs, df_mof)

    #  plt.show()
    plt.savefig("%s/Errors%s_%s_%s" %(RESULT_DIR, prop, val_type, calc_type), dpi=600)
#  plot_error("E", df_data)
def plot_linear_reg_count_norm(val_type, prop, calc_type, df_data):

    #  if single_mof_idx == 1:
    #      NAME_BASE = "mof"
    #      single_mof_idx = 5

    #  mof_name = NAME_BASE+str(single_mof_idx)

    # get single mof data
    #  df_data = df_data.loc[df_data["FileNames"].str.contains(mof_name)]

    plt.rcParams["figure.figsize"] = (16, 10)
    fig, axs = plt.subplots(1, 1)

    if prop == "E":
        axs.set_xlabel("$E_{DFT}$ / eV", size=14)
        axs.set_ylabel("$E_{NNP}$ / eV", size=14)
    if prop == "F" or prop == "FC":
        axs.set_xlabel(r"$F_{DFT}$ / eV$\AA$$^{-1}$", size=14)
        axs.set_ylabel(r"$F_{NNP}$ / eV$\AA$$^{-1}$", size=14)

    all_data_x = []
    all_data_y = []
    all_data_c = []
    for i in range(n_frags):
        frag_base = mof_name+"_f"+str(i+1)
        df_mof = df_data.loc[df_data["FileNames"].str.contains(frag_base)]
        df_mof = df_mof.loc[df_mof["FileNames"].str.contains("co2") == False] # does not contain.
        print("Property %s: %s --> " %(prop, frag_base), len(df_mof))
        if len(df_mof) == 0:
            continue


        if prop == "E":
            x = df_mof.iloc[:, 5].to_numpy()
            y = df_mof.iloc[:, 6].to_numpy()
            #normalizes data
            x = normalizeData(x)
            y = normalizeData(y)
            # shifted data
            #  x = x - x.max()
            #  y = y - y.max()

            all_data_x.append(x)
            all_data_y.append(y)
            all_data_c.append(df_mof.iloc[:, 7].to_numpy())
        elif prop == "FC":
            x = df_mof.iloc[:, 2].to_numpy()
            y = df_mof.iloc[:, 3].to_numpy()
            all_data_x.append(x)
            all_data_y.append(y)
            all_data_c.append(df_mof.iloc[:, 4].to_numpy())

    x = np.concatenate(all_data_x)
    y = np.concatenate(all_data_y)
    m, b = np.polyfit(x, y, 1) # get regression line paremeres
    linreg = stats.linregress(x, y)
    #  axs.plot(x, m * x + b, ls="-", color="k", linewidth=0.8) # plot regression line
    axs.plot([x.min(), x.max()], [y.min(), y.max()], ls="-", color="k", linewidth=0.8) # plot regression line

    axs.text(x.max()-3, y.max(), "$R^2$=%.3f"%linreg.rvalue,
                color="k", ha="right", va="center", size=8)


    c = np.concatenate(all_data_c)
    data_show = axs.scatter(x=x, y=y, s=12,
                      marker="o", alpha=0.5,
                      label=frag_base, c=c, cmap="plasma")
    #  cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    fig.colorbar(data_show)
    axs.legend()

    axs.grid(which='both', axis='both',  linestyle='--', linewidth=0.8)
    #  axs.ticklabel_format(useOffset=False)
    if prop == "E":
        #  axs[i, j].ticklabel_format(style='scientific', axis='both', scilimits=(0, 0))
        axs.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

        axs.xaxis.set_major_locator(ticker.MultipleLocator(20))
        axs.xaxis.set_minor_locator(ticker.MultipleLocator(5))

        if val_type == "test":
           axs.xaxis.set_major_locator(ticker.MultipleLocator(4))
           axs.xaxis.set_minor_locator(ticker.MultipleLocator(1))

        axs.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        axs.yaxis.set_major_locator(ticker.MultipleLocator(20))
        axs.yaxis.set_minor_locator(ticker.MultipleLocator(5))

        if val_type == "test":
           axs.yaxis.set_major_locator(ticker.MultipleLocator(4))
           axs.yaxis.set_minor_locator(ticker.MultipleLocator(1))


    #  plt.show()
    plt.savefig("%s/%s_lineerRegressionCountNorm%s_%s_%s" %(RESULT_DIR, mof_name, prop, val_type, calc_type), dpi=600)

def _plot_linear_reg_count(i, j, fig, ax, df_mof, idx):

    if prop == "E":
        x = df_mof.iloc[:, 5]
        y = df_mof.iloc[:, 6]
    elif prop == "FC":
        x = df_mof.iloc[:, 2]
        y = df_mof.iloc[:, 3]

    x_max = x.max()
    x_min = x.min()
    y_max = y.max()
    y_min = y.min()

    data_range_x = abs(x_max- x_min)
    data_range_y = abs(y_max- y_min)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(data_range_x / 5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(data_range_x / 20))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(data_range_y / 5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(data_range_y / 20))
    m, b = np.polyfit(x, y, 1) # get regression line paremeres
    linreg = stats.linregress(x, y)
    ax.plot(x, m * x + b, ls="-", color="k", linewidth=0.8) # plot regression line

    ax.text(x_max-0.15*data_range_x, y_max-0.03*data_range_y, "$R^2$=%.3f"%linreg.rvalue,
                color="k", ha="right", va="center", size=8)


    #  c = x - y # error
    #  xy = x - y # error
    #  xy = np.vstack([x,y])
    #  c =  gaussian_kde(xy)(xy)

    ax.scatter(x=x, y=y, s=10, marker="o", alpha=0.3, label="F%d" %(idx + 1))
    #  fig.colorbar(data_show, ax=ax, label="Counts")
    ax.legend(prop={'size': 8})

    ax.grid(which='both', axis='both',  linestyle='--', linewidth=0.8)

    return fig, ax

def plot_linear_reg_count(val_type, prop, calc_type, df_data):


    #  if single_mof_idx == 1:
    #      NAME_BASE = "mof"
    #      single_mof_idx = 5
    #  else:

    #      NAME_BASE = "irmofseries"

    dim1 = 3
    dim2 = 3

    plt.rcParams["figure.figsize"] = (18, 12)
    #  fig, axs = plt.subplots(dim1, dim2, sharey=False) # created subplots which share y axis
    fig = plt.figure()

    if prop == "E":
        fig.text(0.5, 0.04, "$E_{DFT}$ / eV $atom^{-1}}$", ha="center", va="center", size=14)
        fig.text(0.05, 0.5, "$E_{NNP}$ / eV $atom^{-1}$", ha="center", va="center", rotation=90, size=14)
    if prop == "F" or prop == "FC":
        fig.text(0.5, 0.04, r"$F_{DFT}$ / eV$\AA$$^{-1}$", ha="center", va="center", size=14)
        fig.text(0.05, 0.5, r"$F_{NNP}$ / eV$\AA$$^{-1}$", ha="center", va="center", rotation=90, size=14)

    idx = 0
    mof_nums = [4, 6, 7, 10] # mof numbers exclude IRMOF-1
    for i in range(dim1):
        for j in range(dim2):
            ax = fig.add_subplot(dim1, dim2, idx+1) # , projection='scatter_density')

            #  get single mof data
            if idx <= 4:
                mof_name = "mof5"
                df_mof = df_data.loc[df_data["FileNames"].str.contains(mof_name)]

                frag_base = "_f" + str(idx+1)
                df_mof = df_mof.loc[df_mof["FileNames"].str.contains(frag_base)]
                #  df_mof = df_mof.loc[df_mof["FileNames"].str.contains("co2") == False] # does not contain.
                print("Property %s: %s%s --> " %(prop, mof_name, frag_base), len(df_mof))
                fig, ax = _plot_linear_reg_count(i, j, fig, ax, df_mof, idx)
                idx += 1
            else:
                #  for mof_num in IDX_MOFs[1:]:
                mof_num = mof_nums.pop(0)
                mof_name = "irmofseries" + str(mof_num)
                df_mof = df_data.loc[df_data["FileNames"].str.contains(mof_name)]

                #  frag_base = "_f" + str(idx+1)
                frag_base = "_f6"
                df_mof = df_mof.loc[df_mof["FileNames"].str.contains(frag_base)]
                #  df_mof = df_mof.loc[df_mof["FileNames"].str.contains("co2") == False] # does not contain.
                print("Property %s: %s%s --> " %(prop, mof_name, frag_base), len(df_mof))

                fig, ax = _plot_linear_reg_count(i, j, fig, ax, df_mof, idx)
                idx += 1

    #  plt.show()
    #  fig.colorbar(data_show, ax=axs)
    #  plt.savefig("%s/%s_lineerRegressionCount%s_%s_%s" %(RESULT_DIR, mof_name, prop, val_type, calc_type), dpi=600)
    plt.savefig("%s/lineerRegression%s_%s_%s" %(RESULT_DIR, prop, val_type, calc_type), dpi=600)


def plot_histograms(val_type, prop, calc_type, df_data):

    colors = ["c", "lightblue", "aqua", "orange", "chocolate", "purple"]

    plt.rcParams["figure.figsize"] = (6, 6)
    fig, axs = plt.subplots()

    # set common x and y axis label
    if prop == "E":
        axs.set_xlabel("$E_{NNP}$ - $E_{DFT}$ (eV $atom^{-1})$", size=12)
        axs.set_ylabel("Frequency", size=12)
    if prop == "F" or prop == "FC":
        axs.set_xlabel(r"$F_{NNP}$ - $F_{DFT}$ (eV $\AA$$^{-1})$", size=12)
        axs.set_ylabel("Frequency", size=12)

    axs.grid(which='both', axis='both',  linestyle='--', linewidth=0.8)
    axs.ticklabel_format(useOffset=False)

    if prop == "E":
        data = df_data.iloc[:, 7]
        axs.hist(data, bins=25, density=True, histtype="barstacked", range=(-0.01, 0.01))
    if prop == "F" or prop == "FC":
        data = df_data.iloc[:, 4]
        axs.hist(data, bins=25, density=True, histtype="barstacked", range=(-0.5, 0.5))
    #  ax.legend()

    # for the normal distribution plot
    mu, std = norm.fit(data)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    axs.plot(x, p, 'k', linewidth=1)

    #  plt.show()
    #  plt.savefig("%s/%s_Histograms%s_%s_%s" %(RESULT_DIR, mof_name, prop, val_type, calc_type), dpi=600)
    #  plt.savefig("%s/%s_Histograms%s_%s_%s" %(RESULT_DIR, frag_name, prop, val_type, calc_type), dpi=600)
    #  plt.savefig("%s_Histograms%s_%s_%s" %(mof_name, prop, val_type, calc_type), dpi=600)
    plt.savefig("%s/HistogramsDensity%s_%s_%s" %(RESULT_DIR, prop, val_type, calc_type), dpi=600)
    plt.close()

if __name__ == "__main__":

    IDX_MOFs = [1, 4, 6, 7, 10]
    BASE_DIR = "/truba_scratch/yzorlu/deepMOF_dev"
    #  model_type = "schnet_l3_basis96_filter64_interact3_gaussian20_rho001_lr00001_bs1_cutoff_60_withoutStress_aseEnv_IRMOFseries1_4_6_7_10_merged_173014_ev"
    model_type = "results_best_epoch_66"
    NAME_BASE = "irmofseries"
    n_frags = 9



    i = 0
    for calc_type in [ "", ]: #"Ensemble",]:
        for val_type in ["test"]:
            for prop in ["FAll"]: # "F"
                dfs = []
                #  for single_mof_idx in IDX_MOFs:
                    #  RESULT_DIR = "%s/schnetpack/results/IRMOF%s/%s" % (BASE_DIR, single_mof_idx, model_type)
                RESULT_DIR = "%s/n2p2/works/runTest/%s" % (BASE_DIR, model_type)
                csv_file_name = "%s/qm_sch_SP%s_%s_%s.csv" %(RESULT_DIR, calc_type, prop, val_type)
                #  column_names = ["FileNames", "qm_SP_energies", "schnet_SP_energies", "Error", "ErrorPerAtom"]
                df_data = pd.read_csv(csv_file_name)
                    #  dfs.append(df_data)

                #  df_data = pd.concat(dfs)
                #  plot_linear_reg_count(val_type, prop, calc_type, df_data)
                #  plot_error_norm(val_type, prop, calc_type, df_data)
                plot_error_norm_density(val_type, prop, calc_type, df_data)
                #  plot_histograms(val_type, prop, calc_type, df_data)
                i += 1


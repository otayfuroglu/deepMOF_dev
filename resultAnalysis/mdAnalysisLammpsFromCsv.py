
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


import matplotlib
matplotlib.use("Agg")

sns.set_context("paper", rc={"grid.linewidth": 0.8})
sns.set_style("ticks", rc={"grid.linestyle": "--"})


def add_curve(df, ax, label):
    ax.plot(df[0], df[1], label=label)
    return ax


def plot_loading(data, n_frame_atoms):
    # Read the loading

    plt.rcParams["figure.figsize"] = (12, 6)  # set figure size
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #  ax.set_ylim(0, 0.1)
    ax.set_xlabel("Pressure (Bar)", size=12)
    ax.set_ylabel("Gas Uptale (wt%)", size=12)

    for df in dfs:
        add_curve(df, ax, label="file_name")

    plt.tight_layout()
    plt.savefig("%s_loading.png" %log_base)
    #  plt.show()


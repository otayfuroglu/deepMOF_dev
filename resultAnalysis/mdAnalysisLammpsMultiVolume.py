#
import lammps_logfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

import matplotlib
matplotlib.use("Agg")

sns.set_context("paper", rc={"grid.linewidth": 0.8})
sns.set_style("ticks", rc={"grid.linestyle": "--"})


import argparse

colors = ["k",  "b", "midnightblue", "darkred", "firebrick", "b", "r", "dimgray", "orange", "m", "y", "g", "c"]




Trange = [50, 100, 200, 300, 400]
step_size = 0.2
P = 60000

plt.figure(figsize=(8, 4))
for T in Trange:
    print(T)
    log_path = f"nnp_train_on26kdata_npt_{T}K_poslow19_{int(P)}Bar/alanates_{int(P)}Bar_{T}K.log"
    log_base = os.path.basename(log_path).split(".")[0]
    log = lammps_logfile.File(log_path)
    data = pd.DataFrame()
    for label in log.get_keywords():
        data[label] = log.get(label)

    data = data[:int(1e5)]
    volume = data["Volume"]
    time_axis = data["Step"] * step_size
    plt.plot(time_axis, volume, marker='.', markersize=2, alpha=1, label=f"{T}K_{P}bar")
    #  plt.plot(time_axis, volume_mean, label=r"Volume (avg.)")

plt.ylabel(r"Volume ($\mathring{A}^3$)")
plt.xlabel('t (fs)')
plt.legend()
plt.tight_layout()
plt.savefig(f"volume_{P}_bar")

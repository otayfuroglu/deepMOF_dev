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

#  parser = argparse.ArgumentParser(description="Give something ...")
#  parser.add_argument("-log", type=str, required=True)
#  parser.add_argument("-stepsize", type=float, required=True)
#  parser.add_argument("-skip", type=int, required=True)
#  args = parser.parse_args()


step_size = 0.5 # fs
skip = 0

#  colors = ["k",  "b", "midnightblue", "darkred", "firebrick", "b", "r", "dimgray", "orange", "m", "y", "g", "c"]

plt.figure(figsize=(8, 4))
plt.ylabel('Number of Molecule')
plt.xlabel('Time (fs)')

BASE_DIR = "/Users/omert/Desktop/deepMOF_dev/n2p2/works/runMD/"
for prefix in ["", "classic"]:
    lammps_log_path = f"{BASE_DIR}/{prefix}flexAdorpsionH2onIRMOF10inNPT_77K/IRMOF10_H2_100Bar_77K.log"
    log_base = os.path.basename(lammps_log_path).split(".")[0]

    log = lammps_logfile.File(lammps_log_path)

    if prefix == "classic":
        prefix = "UFF"
        print("\n", prefix)
        color = "r"
    else:
        prefix = "HDNNP"
        print("\n", prefix)
        color = "b"

    data = pd.DataFrame()
    for label in log.get_keywords():
        data[label] = log.get(label)


    n_frame_atoms = data["Atoms"][0]
    # Read the loading
    n_loading = data["Atoms"].subtract(n_frame_atoms)
    n_loading.to_csv(f"{prefix}_{log_base}loading.csv")

    # Compute the cumulative mean
    #  n_loading_mean = np.cumsum(n_loading) / (np.arange(len(n_loading))+1)
    # Get the time axis
    time_axis = data["Step"] * step_size

    plt.plot(time_axis, n_loading, color=color, label=prefix)
    #  plt.plot(time_axis, n_loading_mean, label='Number of Molecule (avg.)')
plt.legend()
plt.tight_layout()
plt.savefig(f"{log_base}_loading.png", dpi=1000)
    #  plt.show()


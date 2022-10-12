
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import argparse


import matplotlib
#  matplotlib.use("Agg")

sns.set_context("paper", rc={"grid.linewidth": 0.8})
sns.set_style("ticks", rc={"grid.linestyle": "--"})


maker_types = ["v", "8", "s", "o", "*"]
plt.rcParams["figure.figsize"] = (12, 6)  # set figure size
fig = plt.figure()
ax = fig.add_subplot(111)
#  ax.set_ylim(0, 0.1)
ax.set_xlabel("Pressure (Bar)", size=12)
ax.set_ylabel("Gas Uptake (wt%)", size=12)

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-csv_dir", type=str, required=True)
args = parser.parse_args()

csv_dir = args.csv_dir
for file_name in os.listdir(csv_dir):
    if file_name.endswith("csv"):
        print(file_name)
        df = pd.read_csv(f"{csv_dir}/{file_name}").astype(float).sort_values(by=["Pressure"])
        ax.plot(df.Pressure, df.AvgGasMassesPercent,
                linestyle='-', marker=maker_types.pop(),
                label=file_name.replace(".csv", "")
               )
        ax.legend()
        #  break

#  plt.tight_layout()
plt.savefig(f"{csv_dir}/loading.png")
#  plt.show()


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import pandas as pd

from matplotlib.ticker import (MaxNLocator,
                               ScalarFormatter,
                               MultipleLocator,
                               FormatStrFormatter)
import matplotlib.ticker as ticker




sns.set(style="white", color_codes=True)

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-csv_path", type=str, required=True)
args = parser.parse_args()
csv_path = args.csv_path

df_data = pd.read_csv(csv_path)
x = df_data.iloc[:, 1].astype(float).to_list()
y = df_data.iloc[:, 2].astype(float).to_list()
# Create a joint plot with hexbin and marginal histograms
#  plt.plot(x,y, "*")

#  plt.savefig(f"{csv_path.split('.')[0]}.png")
g = sns.jointplot(
    x=x,
    y=y,
    kind="reg",  # Use hexbin plot
    line_kws=dict(color="slategray", ),
    color="skyblue", marker='o', scatter_kws=None,
    height=10,           # Increase the overall figure height (in inches)
    ratio=6,            # Adjust the ratio of the joint axes to the marginal
    #  gridsize=100,
    #  cmap="Blues",
    marginal_kws={"bins": 20, "color": "grey", "linewidth":1},  # Histogram settings
    marginal_ticks=False,

)

# Add identity line (y = x)
#  plt.plot([min(x), max(x)], [min(y), max(y)], 'r--', lw=2)

# Labels and title
#  g.set_axis_labels("DFT [eV/atom]", "NNP [eV/atom]")
g.set_axis_labels(r"$E_{DFT}$ [eV $atom^{-1}$]", r"$E_{NNP}$ [eV $atom^{-1}$]", size=30)
#  g.set_axis_labels(r"$F_{DFT}$ [eV $\AA$$^{-1}$]", r"$F_{NNP}$ [eV $\AA$$^{-1}$]", size=30)
#  plt.suptitle("Correlation Plot with Marginal Histograms", y=1.02)  # Title above the plot

# Make tick markers sparse on the main plot
g.ax_joint.xaxis.set_major_locator(MaxNLocator(nbins=4))
g.ax_joint.yaxis.set_major_locator(MaxNLocator(nbins=4))


#  # Format axis labels to avoid scientific notation
#  formatter = ScalarFormatter(useMathText=True)
formatter = ScalarFormatter(useOffset=False)
formatter.set_scientific(False)
g.ax_joint.xaxis.set_major_formatter(formatter)
g.ax_joint.yaxis.set_major_formatter(formatter)

g.ax_joint.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
g.ax_joint.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

g.ax_joint.tick_params(labelsize=28, )


#  plt.setp(g.ax_marg_y.patches, color="r")
plt.savefig(f"{csv_path.split('.')[0]}.png", bbox_inches='tight')
#  plt.show()

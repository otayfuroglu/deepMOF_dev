import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import math

class SymmetryFunctionData:
    def __init__(self):
        self.parameters = {}
        self.histogram = []

def readSymmetryFunctionHistogram(file_name):
    sfd = SymmetryFunctionData()
    f = open(file_name, "r")
    for line in f:
        split_line = line.split()
        if split_line[0] == "#SFINFO":
            sfd.parameters[split_line[1]] = split_line[2]
        elif split_line[0][0] == "#":
            continue
        else:
            sfd.histogram.append([float(split_line[0]),
                                  float(split_line[1]),
                                  float(split_line[2])])
    f.close()
    sfd.histogram = np.array(sfd.histogram).T
    return sfd

sf_histograms = []
for file_name in sorted(glob.glob("../works/runTrain/weighted_rho10_zeta16_r30a6_l2n30_scalinkMaxMinSF_subRefE_shift_center_gastegger_core24_prunedRMSD/sf.*.histo")):
    sf_histograms.append(readSymmetryFunctionHistogram(file_name))

for sfh in sf_histograms:
    print(sfh.parameters["type"], sfh.parameters["index"])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    X = (sfh.histogram[0] + sfh.histogram[1]) / 2.
    ax.plot(X, sfh.histogram[2])
    mean = float(sfh.parameters["mean"])
    sigma = float(sfh.parameters["sigma"])
    ax.plot([mean - sigma, mean + sigma], [0, 0], "-", lw=3)
    ax.plot(mean, 0, "o")
    ax.plot(float(sfh.parameters["min"]), 0, "o")
    ax.plot(float(sfh.parameters["max"]), 0, "o")

plt.savefig("testSF.png")

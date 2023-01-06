#
import numpy as np
import matplotlib.pyplot as plt


def normal_dist(x, mean, sd):
    #  return (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return 1 / (sd * np.sqrt((2*np.pi))) * np.exp(-0.5*((x-mean)/sd)**2)

def sym_func_g2(rij, rc, rs, n):
    #  return (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    if rc < rij:
        return 0.0
    return  0.5 * (np.cos(np.pi*rij/rc)+1) * np.exp(-n*(rij-rs)**2)

rijs = np.linspace(0, 12, 500)
rc = 12
rs = 0


#  for n in [0.0, 0.006, 0.014, 0.044, 0.079]:
#  for n in [0.0, 0.002, 0.005, 0.007, 0.010, 0.013]:
for n in [0.0, 0.008, 0.019, 0.038, 0.077, 0.177]:
    dist = [sym_func_g2(rij, rc, rs, n) for rij in rijs]
    plt.plot(rijs, dist)
plt.xlabel('Data points')
plt.ylabel('Probability Density')
plt.show()


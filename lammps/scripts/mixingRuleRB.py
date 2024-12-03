#  from ase import units
import math

#  u = units.create_units("2014")

# UFF paramters epsilon(energy units, here kcal/mol) sigma (distance units, here units Angstrom
params = {
    "Mg": [0.1109993, 2.69141],
    "C": [0.104999, 3.43085],
    "O": [0.0599996, 3.11815],
    "H": [0.04399974255, 2.57113],
    "C_CO2": [0.0536541755, 2.8000],
    "O_CO2": [0.156988143, 3.050],
}
def mixingRB(sym1, sym2):
    mixed_epsilon = math.sqrt(params[sym1][0] * params[sym2][0])
    mixed_sigma = (params[sym1][1] + params[sym2][1]) / 2
    return(mixed_epsilon, mixed_sigma)


# for  data.MgF1_charged_bigcell
#  atom_types = {
#
#      1: "C",
#      2: "O",
#      3: "O",
#      4: "Mg",
#      5: "O",
#      6: "H",
#      7: "C",
#      8: "C_CO2",
#      9: "O_CO2",
#  }
#

# for data.MgF1_charged_bigcell
#  atom_types = {
#
#      1: "C",
#      2: "O",
#      3: "O",
#      4: "Mg",
#      5: "C",
#      6: "O",
#      7: "H",
#      8: "C_CO2",
#      9: "O_CO2",
#  }

# for data.MgF2_ddec_bigcell_withCO2

atom_types = {
 1: "C",
 2: "O",
 3: "H",
 4: "O",
 5: "Mg",
 6: "C",
 7: "O",
 8: "C",
 9: "O",
10: "O",
11: "C",
12: "O_CO2",
13: "C_CO2",
}

n_atomstypes = len(atom_types.keys())
j =13
for i in range(1, n_atomstypes+1):
    mixed_epsilon, mixed_sigma = mixingRB(atom_types[i], atom_types[j])
    print(f"pair_coeff {i}  {j}  {mixed_epsilon} {mixed_sigma} # {atom_types[i]}-{atom_types[j]}")

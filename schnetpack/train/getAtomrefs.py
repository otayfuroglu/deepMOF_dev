#
# this file iclude atomization enerhgy per atom in eV unit
import numpy as np

n_elements = 118
atomizationEsIneV = {
    1: -13.568416326526803,
    6: -1027.290084166281,
    8: -2038.792640354206,
    30: -48405.07779371168
}

atomizationEsInAU = {
    1: -0.4986298692884847,
    6: -37.75219657637893,
    8: -74.92421247265553,
    30: -1778.8529650280634,
} # Hartrees Energy

atomizationEsInKcal = {
    1: -312.89511614561894,
    6: -23689.872308239435,
    8: -47015.6755695421,
    30: -1116247.6204898471,
}

def atomrefs_energy0(energy_keyword, unit):
    idx_atomizationE = np.zeros((n_elements, 1))

    if unit == "ev":
        atomizationEs = atomizationEsIneV
    elif unit == "au":
        atomizationEs = atomizationEsInAU
    elif unit == "kcal":
        atomizationEs = atomizationEsInKcal
    else:
        print("Not found atom referans energy in %s unit" %unit)
        exit(1)

    for idx, singleEnergy in atomizationEs.items():
        idx_atomizationE[idx] = np.array(singleEnergy, dtype=np.float64)
    return {energy_keyword: idx_atomizationE}

#  to test
#  print(atomrefs_energy0("energy", "au"))

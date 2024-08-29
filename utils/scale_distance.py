

from ase.io import read, write

atoms = read("./opt_MgF2_CO2_1_bonded.xyz")
indices = [atom.index for atom  in atoms]
#  print(indices)

for distance in range(8, 40):
    atoms.set_distance(18, 81, distance=distance/10, indices=[81, 82, 83])
    write("test.extxyz", atoms, append=True)





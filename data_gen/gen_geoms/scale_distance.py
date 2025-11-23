

from ase.io import read, write
import argparse



parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-geom_path", type=str, required=True,)
args = parser.parse_args()

geom_path = args.geom_path
base =  geom_path.split("/")[-1].split(".")[0]

atoms = read(geom_path)
#  atoms.set_distance(126, 544, distance=1.7, indices=[544, 545, 546])
#  atoms.set_angle(126, 544, 545, angle=145, indices=[544, 545, 546])
#  write("MgMOF74_clean_fromCORE_addedH_withCO2_final.extxyz", atoms)
#  quit()
indices = [atom.index for atom  in atoms]
#  print(indices)

for scale in range(5, 40, 1):
    #  distance = distance/10
    scale = scale/10
    distance = round(atoms.get_distance(72, 108) * scale, 1)
    print(distance)
    atoms_scale = atoms.copy()
    #  atoms_scale.set_distance(18, 81, distance=distance, indices=[81, 82, 83])
    #  atoms_scale.set_distance(, 83, distance=distance, fix=0, indices=[83, 163, 82])
    atoms_scale.set_distance(72, 108, distance=distance, fix=0, indices=[106, 107, 108])
    atoms_scale.info["label"] = f"{base}_scaled_{distance}"
    write(f"{base}_scaled.extxyz", atoms_scale, append=True)





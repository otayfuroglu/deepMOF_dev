#
from ase.io import read
import argparse
import tqdm



parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-extxyz_path", type=str, required=True, help="")
args = parser.parse_args()


file_base = args.extxyz_path.split("/")[-1].split(".")[0]
atoms_list = read(args.extxyz_path, index=":")


with open(f"{file_base}_energeis.csv", "w") as fl_enegies:
    fl_enegies.write(f"StrucIdx,Name,NNP_Energy\n")
    for atoms in tqdm.tqdm(atoms_list):
        model_energy = atoms.get_potential_energy()
        label = atoms.info["label"]
        fl_enegies.write(f"{label.split('_')[1]},{label},{model_energy}\n")
        fl_enegies.flush()

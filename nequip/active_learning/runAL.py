#

from nequip.ase import NequIPCalculator
from ase.io import read, write
from ase.io.trajectory import Trajectory
import tqdm
import argparse

device = "cuda"

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-trj_path", type=str, required=True, help="")
parser.add_argument("-model2_path", type=str, required=True, help="")
parser.add_argument("-version", type=str, required=True, help="")
args = parser.parse_args()

version = args.version

model2_path = args.model2_path
calc2 = NequIPCalculator.from_deployed_model(
    model_path=model2_path,
    device=device,
)


trj_path = args.trj_path
file_base = trj_path.split("/")[-1].split(".")[0]

atoms_list = Trajectory(trj_path)
n_atoms_list = len(atoms_list)

n_sample = 0
#  i = 500
i = 0

fl = open(f"{file_base}_model1_model2_energeis.csv", "w")
fl.write(f"index,e_NNP1,e_NNP2,e_diff\n")

#  while n_sample <= 250:
    # start from middle of MD
for i in tqdm.trange(int(n_atoms_list/2), len(atoms_list), 2):
#  for i in tqdm.trange(0, len(atoms_list), 1):
    atoms = atoms_list[i]
    e1 = atoms.get_potential_energy() / len(atoms)
    atoms.calc = calc2
    e2 = atoms.get_potential_energy() / len(atoms)
    e_diff = abs(e2 - e1)

    fl.write(f"{i},{e1},{e2},{e_diff}\n")
    fl.flush()

    if e_diff >= 0.0002:
        if n_sample < 7:
            atoms.info["label"] = f"{file_base}_{version}_" + "{0:0>5}".format(i)
            #  write(f"all_AL_{version}.extxyz", atoms, append=True)
            write(f"all_AL_{version}_flex_guest.extxyz", atoms, append=True)

            n_sample += 1
fl.close()

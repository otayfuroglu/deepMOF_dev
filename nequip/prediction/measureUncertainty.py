#

from nequip.ase import NequIPCalculator
from ase.io import read, write
from ase.io.trajectory import Trajectory
import tqdm
import argparse
import math


def getUncertainty(atoms_list, calc1, calc2):

    N = len(atoms_list)
    e_diffs = []
    ms = 0.0
    for i in tqdm.trange(0, N):
        #  for i in range(0, len(atoms_list), 1):
        atoms = atoms_list[i]
        atoms.calc = calc1
        e1 = atoms.get_potential_energy() / len(atoms)
        atoms.calc = calc2
        e2 = atoms.get_potential_energy() / len(atoms)
        e_diff = abs(e2 - e1)
        ms += e_diff * e_diff

    return math.sqrt(ms)



parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-trj_path", type=str, required=True, help="")
args = parser.parse_args()

#  file_base = "MgF1"
#  T = 300

device = "cuda"
trj_path = args.trj_path
file_base = trj_path.split("/")[-1].split(".")[0]

atoms_list = read(trj_path, index=":")

fl = open(f"uncertainty_model1_model2_energeis.csv", "w")
fl.write(f"version,rmse\n")

for i in range(1, 6):
    version = "v" + str(i)


    model1_path = f"/truba_scratch/otayfuroglu/deepMOF_dev/nequip/works/mof74/runTrain/results/MgF1_nnp1/{version}/MgF1_{version}_nnp1.pth"
    model2_path = f"/truba_scratch/otayfuroglu/deepMOF_dev/nequip/works/mof74/runTrain/results/MgF1_nnp2/{version}/MgF1_{version}_nnp2.pth"

    calc1 = NequIPCalculator.from_deployed_model(
        model_path=model1_path,
        device=device,
    )
    calc2 = NequIPCalculator.from_deployed_model(
        model_path=model2_path,
        device=device,)

    uncert = getUncertainty(atoms_list, calc1, calc2)
    fl.write(f"{version}, {uncert}\n")


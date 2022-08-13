from ase.io import read
from ase.units import kJ
from ase.eos import EquationOfState

import argparse
parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-traj_path", type=str, required=True, help="..")
args = parser.parse_args()

configs = read("%s@0:" % args.traj_path)
# Extract volumes and energies:
volumes = [ag.get_volume() for ag in configs]
energies = [ag.get_potential_energy() for ag in configs]

eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()

print("V0", v0)
print(B / kJ * 1.0e24, 'GPa')
#  eos.plot('EOS.png')

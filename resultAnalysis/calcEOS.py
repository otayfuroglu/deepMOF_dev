#
from ase.io import read
from ase import eos

import argparse
import os


#  a = 4.0  # approximate lattice constant
#  b = a / 2
#
#  eos = eos.calculate_eos(a, trajectory='_poslow02_SG02_EOS.traj')
#  v, e, B = eos.fit()
#  a = (4 * v)**(1 / 3.0)
#  print('{0:.6f}'.format(a))

def calcEqueConstant():
    configs = read(f"{traj_path}@0:-1")
    if len(configs) == 0:
        return
    volumes = [ag.get_volume() for ag in configs]
    energies = [ag.get_potential_energy() for ag in configs]
    eos_ = eos.EquationOfState(volumes, energies, eos="p3")
    v0, e0, B = eos_.fit()
    print(eos_.eos_parameters)
    #  print(B / kJ * 1.0e24, 'GPa')
    #  print(f"{traj_path}, ", "V0:", v0, "B: ", B)
    print(f"{traj_path},", v0 )
    #  eos.plot('_poslow02_SG02_EOS_eos.png')

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-traj_path", type=str, required=True, help="..")
args = parser.parse_args()
    #  nn = MinimumDistanceNN()
traj_path = args.traj_path
#  for traj_path in [fl for fl in os.listdir("./") if ".traj" in fl]:
#  try:
calcEqueConstant()
#  except:
        #  pass


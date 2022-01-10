#! /truba/home/yzorlu/miniconda3/envs/python38/bin/python -u

import os
import sys

from ase.calculators.socketio import SocketClient, SocketIOCalculator
from ase.optimize import BFGS
from ase.io import read

from schnetpack.interfaces import SpkCalculator
from schnetpack.environment import AseEnvironmentProvider
from schnetpack.utils import load_model
from schnetpack import Properties

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cpu"

mof_num = 1
len_data = 100220
BASE_DIR = "/truba_scratch/yzorlu/deepMOF/HDNNP"

model_schnet = load_model(
    "%s/schnetpack/runTraining"
    "/schnet_l3_basis96_filter64_interact3_gaussian20_"
    "rho001_lr00001_bs1_cutoff_60_withoutStress_aseEnv_IRMOFseries%d_merged_%d_ev"
    "/best_model" %(BASE_DIR, mof_num, len_data),
    map_location=device,
)

mol_path = BASE_DIR + "/prepare_data/geomFiles/IRMOFSeries/cif_files/mercury_IRMOF%s.cif" %mof_num
atoms = read(mol_path)

calc_schnet = SpkCalculator(model_schnet, device=device,
                            energy=Properties.energy,
                            forces=Properties.forces,
                            #  collect_triples=True,
                            environment_provider=AseEnvironmentProvider(cutoff=6.0)
                           )

atoms.set_calculator(calc_schnet)
# Create Client
# In this example we use a UNIX socket.  See other examples for INET socket.
# UNIX sockets are faster then INET sockets, but cannot run over a network.
# UNIX sockets are files.  The actual path will become /tmp/ipi_ase_espresso.
#  unixsocket = 'ase_schnet'
port = 31415
host = "localhost"
client = SocketClient(host="", port=port)
# 'unix'
#  client = SocketClient(unixsocket=host)
client.run(atoms)

#  opt = BFGS(atoms, trajectory='opt.traj',
#             logfile='opt.log')
#
#  with SocketIOCalculator(calc_schnet, log=sys.stdout,
#                          unixsocket=unixsocket) as calc:
#      atoms.calc = calc
#      opt.run(fmax=0.05)























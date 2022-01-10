#! /truba/home/yzorlu/miniconda3/bin/python -u

import os
#  import schnetpack as spk
from schnetpack.md import System
from schnetpack.md import MaxwellBoltzmannInit
from schnetpack.md.integrators import VelocityVerlet, NPTVelocityVerlet
from schnetpack.md.calculators import SchnetPackCalculator
from schnetpack.md.calculators import EnsembleSchnetPackCalculator
from schnetpack import Properties
from schnetpack.md import Simulator
from schnetpack.md.simulation_hooks import thermostats, barostats
from schnetpack.md.simulation_hooks import logging_hooks
from schnetpack.md.integrators import RingPolymer, NPTRingPolymer
from schnetpack.md.neighbor_lists import ASENeighborList, TorchNeighborList

from schnetpack.utils import set_random_seed
from schnetpack.md.utils import HDF5Loader

from ase.io import read
import torch
import argparse
import ntpath

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-mol", "--mol_path",
                    type=str, required=True,
                    help="give full molecule path ")
parser.add_argument("-mdtype", "--md_type",
                    type=str, required=True,
                    help="give MD type classical or Ring Polymer (rpmd) MD")
parser.add_argument("-n", "--n_steps",
                    type=int, required=True,
                    help="give number of stepes")
parser.add_argument("-temp", "--temp",
                    type=int, required=True,
                    help="give MD bath temperature")
parser.add_argument("-init_temp", "--init_temp",
                    type=int, required=True,
                    help="give sysyem temperature temperature")
parser.add_argument("-calc_mode", "--calc_mode",
                    type=str, required=True,
                    help="give md ore active mode")
parser.add_argument("-MODEL1_DIR", "--MODEL1_DIR",
                    type=str, required=True,
                    help="")
parser.add_argument("-MODEL2_DIR", "--MODEL2_DIR",
                    type=str, required=False,
                    help="")
parser.add_argument("-descrip_word", "--descrip_word",
                    type=str, required=True,
                    help="")
parser.add_argument("-opt", "--opt",
                    type=str, default="yes", required=False,
                    help="")
parser.add_argument("-restart", "--restart",
                    type=str, default="no", required=True,
                    help="")


args = parser.parse_args()

# random seed for numpy and torch
seed = None  # random seed according to time
set_random_seed(seed)

# Gnerate a directory of not present
# Get the parent directory of SchNetPack
molecule_path=args.mol_path
md_type=args.md_type
print(md_type)
n_steps=args.n_steps

# set restart option
restart = False
if args.restart == "yes":
    restart = True

_, name = ntpath.split(molecule_path)
name = name[:-4]
BASE_DIR = "/truba_scratch/yzorlu/deepMOF/HDNNP/schnetpack"

WORKS_DIR = os.getcwd() + "/" + name + "_%dK_" % args.temp + md_type + args.descrip_word
MODELS_DIR = [
    BASE_DIR
    + "/runTraining/"
    + args.MODEL1_DIR,
    #  BASE_DIR
    #  + "/runTraining"
    #  + args.MODEL2_DIR,
]


def optStructure(atoms, model):
    from ase.optimize import BFGS, LBFGS
    from schnetpack.interfaces import SpkCalculator
    from schnetpack.environment import AseEnvironmentProvider

    calculator = SpkCalculator(
        model,
        device=device,
        #  collect_triples=True,
        environment_provider=AseEnvironmentProvider(cutoff=6.0),
        energy=properties[0],
        forces=properties[1],
        #  stress=stress,
        energy_units="eV",
        forces_units="eV/Angstrom",
        stress_units="eV/Angstrom/Angstrom/Angstrom",
    )

    atoms.set_calculator(calculator)
    opt = LBFGS(atoms)
    opt.run(fmax=0.015)

    return atoms


def getAseMol(molecule_path, pbc=True):
    atoms = read(molecule_path)
    if pbc is False:
        atoms.pbc = [0, 0, 0]
    return read(molecule_path)

def makeSpercell(molecule_path, P):
    from ase.build import make_supercell
    mol = read(molecule_path)
    mol.pbc = [True, True, True]
    return make_supercell(mol, P)

def getLastPoseFromHDF5(log_file_path, nsample):
    skip_pose = nsample - 1
    data = HDF5Loader(log_file_path, skip_initial=skip_pose, load_properties=False)
    atoms = Atoms(data.get_property(Properties.Z, mol_idx=0),
                  positions=data.get_property(Properties.R, mol_idx=0)[0] * 10.0) # *10.0 for the hdf5 scaling
    return atoms

def setCalculator(calc_mode, models, properties, stress_handle, cutoff):

    position_conversion = "Angstrom"
    force_conversion = "eV / Angstrom"
    stress_conversion="eV / Angstrom / Angstrom / Angstrom"

    if calc_mode == "ensemble":
        md_calculator = EnsembleSchnetPackCalculator(
            models,
            required_properties=properties,
            force_handle=properties[1],
            neighbor_list=ASENeighborList,
            #  neighbor_list=TorchNeighborList,
            stress_handle=stress_handle,
            cutoff=cutoff,

            position_conversion=position_conversion,
            force_conversion=force_conversion,
            stress_conversion=stress_conversion,
        )
    if calc_mode == "md":
        md_calculator = SchnetPackCalculator(
            models[0],
            required_properties=properties,
            force_handle=properties[1],
            neighbor_list=ASENeighborList,
            #  neighbor_list=TorchNeighborList,
            stress_handle=stress_handle,
            cutoff=cutoff,
            position_conversion=position_conversion,
            force_conversion=force_conversion,
            stress_conversion=stress_conversion,
        )

    return md_calculator

def setSimulator(md_type, system_temperature, ensemble, restart=True):

    bath_temperature = args.temp
    target_pressure = 1.01325 # bar
    time_constant = 100
    buffer_size = 50

    # Here, a smaller timestep is required for numerical stability
    if md_type == "rpmd":
        time_step = 0.2  # fs
        print("set of the RingPlymer Integrator")
        # Initialize the thermostat
        if ensemble == "NVT":
            termos = thermostats.PILELocalThermostat(bath_temperature,
                                                     time_constant)
            # termos = thermostats.TRPMDThermostat(bath_temperature,
            # termos = thermostats.NHCRingPolymerThermostat(bath_temperature,
            # time_constant)

            integrator = RingPolymer(
                n_replicas,
                time_step,
                system_temperature,
                device=device
            )

        elif ensemble == "NVE":

            integrator = RingPolymer(
                n_replicas,
                time_step,
                system_temperature,
                device=device
            )
        elif ensemble == "NPT":
            termos = thermostats.PILELocalThermostat(bath_temperature,
                                                     time_constant)
            #  termos = thermostats.NHCRingPolymerThermostat(bath_temperature,
            #                                                time_constant)

            baros = barostats.RPMDBarostat(target_pressure,
                                           system_temperature,
                                           time_constant)
            integrator = NPTRingPolymer(
                n_replicas,
                time_step,
                system_temperature,
                baros,
                device=device,
            )

    elif md_type == "md":
        time_step = 0.2  # fs
        if ensemble == "NVT":
            #  print("set of the VelocityVerlet Integrator")
            termos = thermostats.LangevinThermostat(bath_temperature,
                                                    time_constant)
            integrator = VelocityVerlet(time_step)
        elif ensemble == "NVE":
            #  print("set of the VelocityVerlet Integrator")
            integrator = VelocityVerlet(time_step)
        elif ensemble == "NPT":
            termos = thermostats.LangevinThermostat(bath_temperature,
                                                    time_constant)
            baros = barostats.NHCBarostatIsotropic(target_pressure,
                                                   bath_temperature,
                                                   time_constant)
            integrator = NPTVelocityVerlet(time_step, baros)
    else:
        raise NameError("%s is a bad name for MD type" % md_type)

    # Logging
    log_file = os.path.join(WORKS_DIR, '%s_%s.hdf5' % (name, md_type))
    data_streams = [
        logging_hooks.MoleculeStream(),
        logging_hooks.PropertyStream(),
    ]
    file_logger = logging_hooks.FileLogger(
        log_file,
        buffer_size,
        data_streams=data_streams,
        every_n_steps=5,
    )

    # Checkpoints
    chk_file = os.path.join(WORKS_DIR, '%s_%s.chk' % (name, md_type))
    if restart:
        checkpoint = torch.load(chk_file)
    else:
        checkpoint = logging_hooks.Checkpoint(chk_file, every_n_steps=100)

    # Assemble the hooks:
    if ensemble == "NVT":
        simulation_hooks = [termos, file_logger]
    elif ensemble == "NVE":
        simulation_hooks = [file_logger]
    elif ensemble == "NPT":
        simulation_hooks = [termos, baros, file_logger]

    if not restart:
        simulation_hooks += [checkpoint]

    # Assemble the simulator
    simulator = Simulator(
        system,
        integrator,
        md_calculator,
        simulator_hooks=simulation_hooks,
        restart=restart
    )

    if restart:
        simulator.load_system_state(checkpoint)
        #  simulator.restart_simulation(checkpoint)
    return simulator


if __name__ == "__main__":

    if md_type == "rpmd":
        n_replicas = 4
    else:
        n_replicas = 1

    print("# of replicas --> ", n_replicas)

    if not os.path.exists(WORKS_DIR):
        os.mkdir(WORKS_DIR)

    properties = ["energy", "forces", "stress"]  # properties used for training

    # Load model and structure
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #  device = "cpu"
    n_gpus = torch.cuda.device_count()
    print("Number of cuda devices --> %s" % n_gpus,)
    print("Forces Handle from --> ", properties[1])

    models_path = [os.path.join(MODEL_DIR, "best_model") for MODEL_DIR in MODELS_DIR]
    models = [torch.load(model_path, map_location=device).to(device) for model_path in models_path]


    calc_mode = args.calc_mode
    print(calc_mode)

    # set md calculator
    md_calculator = setCalculator(calc_mode, models, properties, stress_handle="stress", cutoff=5.5)

    system_temperature = args.init_temp

    # Set up the system, load structures, initialize
    system = System(n_replicas, device=device)
    #  system.load_molecules_from_xyz(molecule_path)
    #  system.load_molecules(getAseMol(molecule_path, pbc=True))

    P = [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]
    supercell = makeSpercell(molecule_path, P)

    # Geometric optimzataion of structures
    if args.opt == "yes":
        print("Optmization proccess stated for enerji minimization")
        supercell = optStructure(supercell, model=models[0])
    system.load_molecules(supercell)

    # Initialize momenta
    initializer = MaxwellBoltzmannInit(
        system_temperature,
        remove_translation=True,
        remove_rotation=True)

    initializer.initialize_system(system)

    #  simulator = setSimulator(md_type, system_temperature, ensemble="NVT", restart=False)
    #  simulator.simulate(n_steps=1000000)
    simulator = setSimulator(md_type, system_temperature, ensemble="NPT", restart=restart)
    simulator.simulate(n_steps)

    # setting strain for pressure deformation simultaions

    # lattice direction a
    #  abc = supercell.cell.lengths()
    #  print(supercell.cell.lengths())
    #  a = abc[0]
    #  print(a)

    #  for i in range(150):
    #      a = a - 0.003
    #      abc[0] = a
    #      supercell.set_cell(abc)
    #      print(supercell.cell.lengths())
    #      system.load_molecules(supercell)
    #      simulator = setSimulator(md_type, system_temperature, ensemble="NPT", restart=True)
    #      simulator.simulate(n_steps)


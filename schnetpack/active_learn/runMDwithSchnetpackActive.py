#! /truba/home/yzorlu/miniconda3/bin/python -u

import os
#  import schnetpack as spk
from schnetpack.md import System
from schnetpack.md import MaxwellBoltzmannInit
from schnetpack.md.integrators import VelocityVerlet
from schnetpack.md.calculators import SchnetPackCalculator
from schnetpack.md.calculators import SchnetPackCalculatorActive
from schnetpack.md.calculators import EnsembleSchnetPackCalculatorActive
#  from schnetpack import Properties
from schnetpack.md import SimulatorActive
from schnetpack.md.simulation_hooks import thermostats
from schnetpack.md.simulation_hooks import logging_hooks
from schnetpack.md.integrators import RingPolymer
from schnetpack.md.neighbor_lists import ASENeighborList, TorchNeighborList

from schnetpack.utils import set_random_seed
from model_config_param import config

from ase.io import read
import torch
import argparse
import ntpath
import random

# random seed for numpy and torch
seed = None # random seed according to time
set_random_seed(seed)

# Gnerate a directory of not present
# Get the parent directory of SchNetPack
#  BASE_DIR = "/truba_scratch/yzorlu/deepMOF/HDNNP"
#
#  MODELS_DIR = [BASE_DIR + "/schnetpack/runTraining/hdnnBehler_l3n100_rho001_r20a5_lr0001_bs1_IRMOFseries6_merged_84943_ev",
#                BASE_DIR + "/schnetpack/runTraining/hdnnBehler_l3n50_rho001_r20a5_lr0001_bs1_IRMOFseries6_merged_84943_ev",
#               ]
#
parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-mdtype", "--md_type",
                    type=str, required=True,
                    help="give MD type classical or Ring Polymer (rpmd) MD")
parser.add_argument("-n", "--n_steps",
                    type=int, required=True,
                    help="give number of stepes")
parser.add_argument("-temp", "--temp",
                    type=int, required=True,
                    help="give MD bath temperature")
parser.add_argument("-fragName", "--fragName",
                    type=str, required=True,
                    help="give molecule of fragment base name")

parser.add_argument("-maxMinSFPath", "--maxMinSFPath",
                    type=str, required=True,
                    help="give Max Min SF data path")

parser.add_argument("-outOfGeomsDIR", "--outOfGeomsDIR",
                    type=str, required=True,
                    help="give outOfSFGeoms directory")

parser.add_argument("-mode", "--mode",
                    type=str, required=True,
                    help="give md or active mode")

parser.add_argument("-nSamplesPerFragment", "--nSamplesPerFragment",
                    type=int, required=True,
                    help="Enter number of sample per fragment")

parser.add_argument("-threshold", "--threshold",
                    type=float, required=True,
                    help="Enter threshold for differend of NNP1 end NNP2 energies ")

parser.add_argument("-MODEL1_DIR", "--MODEL1_DIR",
                    type=str, required=True,
                    help="")

parser.add_argument("-MODEL2_DIR", "--MODEL2_DIR",
                    type=str, required=True,
                    help="")

args = parser.parse_args()

MODELS_DIR = [args.MODEL1_DIR, args.MODEL2_DIR]

def getAseMol(molecule_path, pbc=True):
    print("Periodic Boundry Condition --> ", pbc)
    mol = read(molecule_path)
    if pbc:
        mol.center(vacuum=5)
        mol.set_pbc((True, True, True))
    else:
        mol.center(vacuum=5)
        mol.set_pbc((False, False, False))
    #  print(mol)
    return mol


def main(molecule_path, md_type, n_steps, n_requiredOutOfGeoms, nOutOfGeoms):

    _, name = ntpath.split(molecule_path)
    name = name[:-4]
    WORKS_DIR = os.getcwd() + "/" + name + "_md_%dK" % args.temp + md_type

    if md_type == "rpmd":
        n_replicas = 8
    else:
        n_replicas = 1

    print("# of replicas --> ", n_replicas)

    if not os.path.exists(WORKS_DIR):
        os.mkdir(WORKS_DIR)

    properties = ["energy", "forces"]#, "stress"]  # properties used for training

    # Load model and structure
    device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print("Number of cuda devices --> %s" % n_gpus,)
    print("Forces Handle from --> ", properties[1])

    models_path = [os.path.join(MODEL_DIR, "best_model") for MODEL_DIR in MODELS_DIR]
    md_models = [torch.load(model_path, map_location=device).to(device) for model_path in models_path]


    position_conversion = "Angstrom"
    force_conversion = "eV / Angstrom"
    mode = args.mode
    if mode == "active":
        md_calculator = SchnetPackCalculatorActive(
            md_models[0],
            required_properties=properties,
            force_handle=properties[1],
            position_conversion=position_conversion,
            force_conversion=force_conversion,
            neighbor_list=TorchNeighborList,
            cutoff=6.0,
            cutoff_shell=1.0,
            detach=True,
            temp=args.temp,
            fragName=args.fragName,
            maxMinSFPath=args.maxMinSFPath,
            outOfGeomsDIR=args.outOfGeomsDIR,
            checkModel=md_models[1], # second model adjusted checkModel
            config=config,
            n_requiredOutOfGeoms=n_requiredOutOfGeoms,
            nOutOfGeoms=nOutOfGeoms,
            threshold=args.threshold,
        )

    if mode == "ensembleActive":
        print(mode)
        md_calculator = EnsembleSchnetPackCalculatorActive(
            md_models,
            required_properties=properties,
            force_handle=properties[1],
            position_conversion=position_conversion,
            force_conversion=force_conversion,
            neighbor_list=TorchNeighborList,
            cutoff=6.0,
            cutoff_shell=1.0,
            detach=True,
            temp=args.temp,
            fragName=args.fragName,
            maxMinSFPath=args.maxMinSFPath,
            outOfGeomsDIR=args.outOfGeomsDIR,
            checkModel=md_models[1], # second model adjusted checkModel
            config = config,
            n_requiredOutOfGeoms=n_requiredOutOfGeoms,
            nOutOfGeoms=nOutOfGeoms,
            threshold=args.threshold,
        )

    # if mode == "md":
    #     md_calculator = SchnetPackCalculator(
    #         md_models[0],
    #         required_properties=properties,
    #         force_handle=properties[1],
    #         position_conversion=position_conversion,
    #         force_conversion=force_conversion,
    #         neighbor_list=TorchNeighborList,
    #         cutoff=6.0,
    #         cutoff_shell=1.0,
    #         detach=True,
    #     )
    system_temperature = 100
    bath_temperature = args.temp
    time_constant = 100
    buffer_size = 100

   # Set up the system, load structures, initialize
    system = System(n_replicas, device=device)
    #  system.load_molecules_from_xyz(molecule_path)
    system.load_molecules(getAseMol(molecule_path, pbc=False))

    # Initialize momenta
    initializer = MaxwellBoltzmannInit(
        system_temperature,
        remove_translation=True,
        remove_rotation=True)

    initializer.initialize_system(system)

    # Here, a smaller timestep is required for numerical stability
    if md_type == "rpmd":
        time_step = 0.5  # fs

        # Initialize the integrator, RPMD also requires
        # a polymer temperature which determines the coupling of beads.
        # Here, we set it to the system temperature
        print("set of the RingPlymer Integrator")
        integrator = RingPolymer(
            n_replicas,
            time_step,
            system_temperature,
            device=device
        )

        # Initialize the thermostat
        termos = thermostats.PILELocalThermostat(bath_temperature,
                                                 time_constant)
        # termos = thermostats.TRPMDThermostat(bath_temperature,
        # termos = thermostats.NHCRingPolymerThermostat(bath_temperature,
        # time_constant)
    else:

        print("set of the VelocityVerlet Integrator")
        time_step = 0.5  # fs
        integrator = VelocityVerlet(time_step)
        termos = thermostats.LangevinThermostat(bath_temperature,
                                                time_constant)

    # Logging
    log_file = os.path.join(WORKS_DIR, md_type+'_%s.hdf5' % name)
    data_streams = [
        logging_hooks.MoleculeStream(),
        logging_hooks.PropertyStream(),
    ]

    #we don't want save any logger during active sampling
    every_n_steps = 10000000
    file_logger = logging_hooks.FileLogger(
        log_file,
        buffer_size,
        data_streams=data_streams,
        every_n_steps=every_n_steps,
    )

    # Checkpoints
    chk_file = os.path.join(WORKS_DIR, md_type+'_%s.chk' % name)
    checkpoint = logging_hooks.Checkpoint(chk_file, every_n_steps=every_n_steps)

    # Assemble the hooks:
    # if md_type == "rpmd":
    simulation_hooks = [
        termos,
        file_logger,
        checkpoint
    ]

    #  else:
    #      simulation_hooks = [
    #          langevin,
    #          file_logger,
    #          checkpoint
    #      ]

    # Assemble the simulator
    simulator = SimulatorActive(
        system,
        integrator,
        md_calculator,
        simulator_hooks=simulation_hooks,
        restart=True,
    )
    simulator.simulate(n_steps)

    return md_calculator.nOutOfGeoms


if __name__ == "__main__":

    nOutOfGeoms = 0
    n_requiredOutOfGeoms = args.nSamplesPerFragment

    nonEquGeomFilesDIR = args.outOfGeomsDIR
    nonEquGeomFrags = [item for item in os.listdir(nonEquGeomFilesDIR) if args.fragName in item]

    while nOutOfGeoms < n_requiredOutOfGeoms:
        i = random.randint(0, len(nonEquGeomFrags) - 1)
        print(nonEquGeomFrags[i])
        try:
            mol_path = "%s/%s" %(nonEquGeomFilesDIR, nonEquGeomFrags[i])
            nOutOfGeoms = main(molecule_path=mol_path, md_type=args.md_type,
                               n_steps=args.n_steps, n_requiredOutOfGeoms=n_requiredOutOfGeoms,
                               nOutOfGeoms=nOutOfGeoms)
            print("\nNumber of Saved Out Of Geometries", nOutOfGeoms, "\n")
        except:

            nOutOfGeoms += 1
            print("MD not start for %s geometry! Skipping..." %nonEquGeomFrags[i])
            print("\nNumber of Saved Out Of Geometries", nOutOfGeoms, "\n")
            pass

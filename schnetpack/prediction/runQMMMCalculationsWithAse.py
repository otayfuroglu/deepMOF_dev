#! /truba/home/yzorlu/miniconda3/bin/python -u
#
from calculationsWithAse import AseCalculations
from schnetpack.environment import AseEnvironmentProvider

#  from schnetpack import Properties
from schnetpack.interfaces import SpkCalculator
from schnetpack.utils import load_model
#  from schnetpack.datasets import AtomsData
import schnetpack

#  from ase import units
from ase.optimize import BFGS, LBFGS, GPMin, QuasiNewton
from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import Trajectory
#  from ase.io.xyz import read_xyz,  write_xyz
#  from ase.io import read, write
from ase.build import molecule

from LammpsInterfacePar import Parameters


import numpy as np
import torch
import os
import shutil
#  import tqdm
#  from writeTestOutput import writeTestOutput

import getpass
USER = getpass.getuser()

properties = ["energy", "forces"]#, "stress"]  # properties used for training
BASE_DIR = f"/truba_scratch/{USER}/deepMOF_dev/"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# in case multiprocesses, global device variable rise CUDA spawn error.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print("Number of cuda devices --> %s" % n_gpus,)


def getWorksDir(calc_name):

    WORKS_DIR = calc_name
    if os.path.exists(WORKS_DIR):
        shutil.rmtree(WORKS_DIR)
    os.mkdir(WORKS_DIR)

    return WORKS_DIR


def getMLcalc():

    MODEL_DIR = BASE_DIR \
            + "/works" \
            + "/runTrain" \
            + "/schnet_l3_basis96_filter64_interact3_gaussian20_rho001_lr00001_bs1_cutoff_60_"\
            + "withoutStress_aseEnv_test_IRMOFseries1_4_6_7_10_merged_173014_ev"
    #  + "/hdnnBehler_l3n100_rho001_r20a5_lr0001_bs1_IRMOFseries%s_merged_72886_ev" %mof_num\

    model_path = os.path.join(MODEL_DIR, "best_model")
    model = load_model(model_path, map_location=device)
    stress = None
    if "stress" in properties:
        print("Stress calculations are active")
        schnetpack.utils.activate_stress_computation(model)
        stress = "stress"

    calculator = SpkCalculator(
        model,
        device=device,
        collect_triples=False,
        environment_provider=AseEnvironmentProvider(cutoff=5.5),
        energy=properties[0],
        forces=properties[1],
        stress=stress,
        energy_units="eV",
        forces_units="eV/Angstrom",
        stress_units="eV/Angstrom/Angstrom/Angstrom",
        )

    return calculator


def getLJcalc():
    from ase.calculators.lj import LennardJones
    from ase.calculators.qmmm import RescaledCalculator

    calculator = LennardJones()
    calculator.parameters.epsilon = 0.00318
    calculator.parameters.sigma = 0.2928
    calculator.parameters.rc = 12.0

    return calculator


def getDFTBcalc():
    from ase.calculators.dftb import Dftb

    calculator = Dftb(
        Hamiltonian_SCC='No',
        Hamiltonian_SCCTolerance=1e-4,
        Hamiltonian_MaxAngularMomentum_='',
        Hamiltonian_MaxAngularMomentum_H='s',
        Hamiltonian_MaxAngularMomentum_C='p',
        Hamiltonian_MaxAngularMomentum_O='p',
        Hamiltonian_MaxAngularMomentum_Zn='p',
        kpts=(1, 1, 1),
    )

    return calculator


def getGPAWcalc():
    from gpaw import GPAW, PW
    return  GPAW(xc='PBE', mode=PW(300))


def getOrcaCalc(n_cpu=40, initial_gbw=["", ""]):
    from ase.calculators.orca import ORCA

    calculator = ORCA(
        maxiter=200,
        charge=0, mult=1,
        orcasimpleinput='SP PBE DEF2-SVP DEF2/J RIJDX MINIPRINT NOPRINTMOS NOPOP NoKeepInts NOKEEPDENS '\
                        + ' ' + initial_gbw[0],
        orcablocks= '%scf Convergence normal \n maxiter 40 end \n %pal nprocs ' + str(n_cpu) + ' end' + initial_gbw[1])
    return calculator


def getEMTcalc():
    from asap3 import EMT
    return EMT()


def getLammpsCalc(WORKS_DIR):
    from ase.calculators.lammpsrun import LAMMPS
    os.environ["ASE_LAMMPSRUN_COMMAND"] = "lmp_serial"
    LAMMPS_POTENTIALS_DIR = os.environ["LAMMPS_POTENTIALS_DIR"]
    parameters = {"pair_style": "comb3 polar_on",
                  "pair_coeff": ["* * ffield.comb3 H C O Zn"],
                  "atom_style": "full",
                  "units": "metal",
                  #  "bond_style": "harmonic",
                  #  "angle_style": "cosine/periodic",
                  #  "dihedral_style": "harmonic",
                  #  "improper_style": "fourier",
                  #  "special_bonds": " lj/coul 0.0 0.0 1.0",
                  #  "dielectric":   "1.0",
                  #  "pair_modify":   "tail yes mix arithmetic",
                  #  "box tilt ":   "large",
                  #  "pair_coeff": ["1 1 0.105 3.43",
                  #                 "2 2 0.044 2.71",
                  #                 "3 3 0.06 3.118",
                  #                 "4 4 0.124 2.462"],
                  #  "kspace_style": "ewald 0.000001",
                  #  "kspace_modify": "gewald 3.0",
                 }
    #  parameters = getParamLammps(BASE_DIR + "/works/runMLMM/in.IRMOF-1")
    #  print(parameters)
    files = [LAMMPS_POTENTIALS_DIR+"/ffield.comb3",
             LAMMPS_POTENTIALS_DIR+"/lib.comb3"
            ]
    for fl in files: shutil.copy(fl, WORKS_DIR)
    print(files)
    #  par_lammps_interface = Parameters()
    lmp = LAMMPS(
        parameters=parameters,
        tmp_dir="./lammpstmp",
        files=files
        #  no_data_file=True,
        )
    return lmp


def getParamLammps(file_path):
    parameters = {}
    with open(file_path) as lines:
        for line in lines:
            if line.strip(" ").startswith("#") or len(line) <= 1:
                continue
            parameters[line.split()[0]] = " ".join(line.split()[1:])
    return parameters


def getLammpsCalc_v2(WORKS_DIR):
    from ase.calculators.lammpslib import LAMMPSlib
    #  from lammps import lammps
    #  lines = open(BASE_DIR+"/works/runMLMM/in.IRMOF-1", 'r').readlines()
    #  cmds = [line.strip("\n") for line in lines if not line.startswith("#") and len(line) > 1]
    cmds = ["pair_style lj/cut 12.500",]
    #  shutil.copy(BASE_DIR+"/works/runMLMM/data.IRMOF-1", WORKS_DIR)
    lammps_header = ["units real", "atom_style full"]
    amendments = ["special_bonds   lj/coul 0.0 0.0 1.0", "dielectric      1.0 ",
                  "pair_modify     tail yes mix arithmetic", 
                 ]

    #  lmp = lammps()
    #  for cmd in cmds:
    #      lmp.command(cmd)
    lammps = LAMMPSlib(lmpcmds=cmds, lammps_header=lammps_header,
                       amendments=amendments, log_file='test.log')
    return lammps

def run(file_base, molecule_path, run_type):

    temp = 300
    #  asePlatformDIR = BASE_DIR + ""
    name = file_base + "_%s_%sK" % (run_type, temp)
    CW_DIR = os.getcwd()

    # main directory for caculation runOpt
    if not os.path.exists("ase_worksdir"):
        os.mkdir("ase_worksdir")

    WORKS_DIR = getWorksDir(BASE_DIR + "/works/runMLMM" + "/ase_worksdir/" + name)

    os.chdir(WORKS_DIR)

    calculation = AseCalculations(WORKS_DIR)
    calculation.setCalcName(name)

    calculation.load_molecule_fromFile(molecule_path)

    n_atoms_bulk = len(calculation.molecule)
    print(n_atoms_bulk)

    #add h2 moecules
    #  calculation.attach_molecule(molecule("H2"), 5, distance=3.5)
    #  P = [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]
    #  calculation.makeSupercell(P)
    #  calculation.molecule.center(vacuum=1.0)

    # qm_selection_musk must be numpy array for ase bug
    # selection of gas
    qm_selection_mask = np.arange(n_atoms_bulk, len(calculation.molecule))
    # selection of frame
    #  qm_selection_mask = np.arange(n_atoms_bulk)
    calculation.save_molecule()

    #  calculation.setQMMMcalculator(
    #      qm_region=qm_selection_mask,
    #      qm_calcultor=getMLcalc(),
    #      mm_calcultor=getDFTBcalc()
    #  )


    #  calculation.setQMMMForceCalculator(
    #      qm_selection_mask=qm_selection_mask,
    #      qm_calcultor=getDFTBcalc(),
    #      mm_calcultor=getMLcalc(),
    #      buffer_width=3.5
    #  )

    #  from ase.calculators.tip4p import TIP4P
    #  calculation.setEIQMMMCalculator(
    #      qm_selection=qm_selection_mask,
    #      qm_calcultor=getOrcaCalc(),
    #      mm_calcultor=TIP4P(),
    #  )

    calculation.molecule.calc = getLammpsCalc(WORKS_DIR)

    if run_type == "opt":
        calculation.optimize(fmax=0.05)
        os.chdir(CW_DIR)

    if run_type == "sp":
        calculation.get_potential_energy()
        os.chdir(CW_DIR)

    elif run_type == "vibration":
        #  WORKS_DIR = WORKS_DIR.replace("vibration", "opt").replace("DFT", "model")
        #  opt_molpath= "%s/Optimization.xyz" %WORKS_DIR
        #  calculation.load_molecule_fromFile(molecule_path=opt_molpath)
        calculation.optimize(fmax=0.015)
        calculation.vibration(nfree=4)
        os.chdir(CW_DIR)

    elif run_type == "md":

        calculation.init_md(
          name=name,
          time_step=0.5,
          temp_init=100.0,
          # temp_bath should be None for NVE and NPT
          temp_bath=temp,
          # temperature_K for NPT
          #  temperature_K=temp,
          pressure = 35,
          interval=5,
        )

        calculation.optimize(fmax=0.5)
        calculation.run_md(1000000)

        #  setting strain for pressure deformation simultaions

        #  lattice direction a
        #  abc = calculation.molecule.cell.lengths()
        #  a = abc[0]

        #  for i in range(149):
        #      print("\nStep %s\n" %i)
        #      a -= 0.003 * a
        #      abc[0] = a
        #      calculation.molecule.set_cell(abc)
        #      calculation.run_md(5000)

    elif run_type == "EOS":

        cell = calculation.molecule.get_cell()


        traj_name = "%s_EOS.traj" % (file_base)
        traj = Trajectory(traj_name, "w")
        scaleFs = np.linspace(0.98, 1.10, 12)
        print(len(scaleFs))
        for scaleF in scaleFs:
            calculation.molecule.set_cell(cell * scaleF, scale_atoms=True)
            calculation.get_potential_energy()

            traj.write(calculation.molecule)
        os.chdir(CW_DIR)


    elif run_type == "optLattice":
        name = file_base + "_calc_lattice_const"
        from ase.constraints import StrainFilter, UnitCellFilter

        #  calculation.setCalcName("mof5Lattice")
        #  calculation.molecule.set_pbc([True, True, True])
        print(calculation.molecule.get_cell_lengths_and_angles())
        calculation.optimizeWithStrain(fmax=0.0005)
        print(calculation.molecule.get_cell_lengths_and_angles())

        calculation.optimize(fmax=0.05)


def main():
    mof_num = 1
    file_base = "IRMOF%s" %mof_num

    MOL_DIR = BASE_DIR + "/geom_files/IRMOFSeries/cif_files"
    molecule_path = os.path.join(MOL_DIR, "%s.cif" %file_base)
    #  molecule_path = BASE_DIR + "/geom_files/methane.xyz"

    run_type = "opt"

    run(file_base, molecule_path, run_type)

main()

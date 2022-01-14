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


def getMLcalculator():

    MODEL_DIR = BASE_DIR \
            + "/works" \
            + "/runTrain" \
            + "/schnet_l3_basis96_filter64_interact3_gaussian20_rho001_lr00001_bs1_cutoff_60_"\
            + "withoutStress_aseEnv_test_IRMOFseries1_4_6_7_10_merged_173014_ev"
    #  + "/hdnnBehler_l3n100_rho001_r20a5_lr0001_bs1_IRMOFseries%s_merged_72886_ev" %mof_num\

    model_path = os.path.join(MODEL_DIR, "best_model")
    model = load_model(model_path, map_location=device)
    if "stress" in properties:
        print("Stress calculations are active")
        schnetpack.utils.activate_stress_computation(model)
    try:
        stress = properties[2]
    except:
        stress = None
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


def getLJcalculator():
    from ase.calculators.lj import LennardJones
    from ase.calculators.qmmm import RescaledCalculator

    calculator = LennardJones()
    calculator.parameters.epsilon = 0.0032
    calculator.parameters.sigma = 0.296
    calculator.parameters.rc = 6.0

    return calculator


def getDFTB():
    from ase.calculators.dftb import Dftb

    calculator = Dftb(
        Hamiltonian_SCC='No',
        Hamiltonian_SCCTolerance=1e-2,
        Hamiltonian_MaxAngularMomentum_='',
        Hamiltonian_MaxAngularMomentum_H='s',
        Hamiltonian_MaxAngularMomentum_C='p',
        Hamiltonian_MaxAngularMomentum_O='p',
    )

    return calculator

def run(file_base, molecule_path, run_type):

    temp = 300
    asePlatformDIR = BASE_DIR + ""
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

    #add h2 moecules
    calculation.attach_molecule(molecule("H2"), 5, distance=3.5)
    #  P = [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]
    #  calculation.makeSupercell(P)

    # qm_selection_musk must be numpy array for ase bug
    qm_selection_mask = np.arange(n_atoms_bulk, len(calculation.molecule))
    print(qm_selection_mask)
    calculation.save_molecule()

    #  calculation.setQMMMcalculator(
    #      qm_region=qm_selection_mask,
    #      qm_calcultor=getDFTB(),
    #      mm_calcultor=getMLcalculator(),
    #  )


    #      qm_selection_mask=qm_selection_mask,
    #      qm_calcultor=getLJcalculator(),
    #      mm_calcultor=getMLcalculator(),
    #      buffer_width=3.5
    #  )

    calculation.setEIQMMMCalculator(
        qm_selection=qm_selection_mask,
        qm_calcultor=getDFTB(),
        mm_calcultor=getMLcalculator(),
    )

    if run_type == "opt":
        calculation.optimize()
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
          interval=5,
        )

        calculation.optimize(fmax=0.9)
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
    file_base = "mercury_IRMOF%s" %mof_num

    MOL_DIR = BASE_DIR + "/geom_files/IRMOFSeries/cif_files"
    molecule_path = os.path.join(MOL_DIR, "%s.cif" %file_base)

    run_type = "md"

    run(file_base, molecule_path, run_type)

main()

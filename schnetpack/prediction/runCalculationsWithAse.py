#
from calculationsWithAse import AseCalculations

#  from schnetpack.data.atoms import AtomsConverter
#  from schnetpack.utils.spk_utils import DeprecationHelper
#  from schnetpack import Properties
#  from schnetpack.interfaces import SpkCalculator
#  from schnetpack.datasets import AtomsData

from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import Trajectory
from ase.io.xyz import read_xyz,  write_xyz
from ase.io import read, write

import time


from ase.optimize import BFGS, LBFGS, GPMin, QuasiNewton

import numpy as np
import os, shutil
#  import tqdm

def getWorksDir(calc_name):

    WORKS_DIR = calc_name
    if os.path.exists(WORKS_DIR):
        shutil.rmtree(WORKS_DIR)
    os.mkdir(WORKS_DIR)

    return WORKS_DIR



#  if os.path.exists("%s.gbw" % file_base):
#      initial_gbw = ['MORead',  '\n%moinp "{}"'.format("initial.gbw")]
#  else:
#      initial_gbw = ["", ""]
#  calculation.setOrcaCalculator("test_%s" %file_base, 40, initial_gbw=initial_gbw)
#  #  print(calculation.calculate_single_point())
#  calculation.optimize(fmax=0.001)
#  calculation.print_calc()

#  start_time = time.time()
#  calculation.setSiestaCalculator(name=file_base, nproc=56)
#  calculation.get_potential_energy()
#  print("Calculation take %s minutes\n" % ((time.time() - start_time) / 60))
#  calculation.print_calc()


def test():
    from writeTestOutput import writeTestOutput
    name = file_base + "_siesta_meshCutoff_test"
    WORKS_DIR = getWorksDir(name)
    CW_DIR = os.getcwd()
    os.chdir(WORKS_DIR)

    start_time = time.time()
    calculation = AseCalculations(WORKS_DIR)
    calculation.load_molecule_fromFile(molecule_path)

    calculation.setSiestaCalculator(name, 40)
    calculation.setCalcName(name)
    potE = calculation.get_potential_energy()

    os.chdir(CW_DIR)
    spend_time = (time.time() - start_time) / 60.0  # in minutes
    print("Calculation take %s minutes\n" % spend_time)
    headers = ["calc_option", "potE_ev", "time"]
    test_outputs = ["meshCutoff_700_k222", potE, spend_time]
    writeTestOutput("result_test_meshCutoff.csv", headers, test_outputs)


#  test()


def run(file_base, molecule_path, calc_type, run_type):

    temp = 300
    name = file_base + "_%s_%s_%sK" % (calc_type, run_type, temp)
    CW_DIR = os.getcwd()

    # main directory for caculation runOpt
    if not os.path.exists("ase_worksdir"):
        os.mkdir("ase_worksdir")

    WORKS_DIR = getWorksDir(BASE_DIR + "/schnetpack/works/runTest/ase_worksdir/" + name)

    os.chdir(WORKS_DIR)

    calculation = AseCalculations(WORKS_DIR)
    calculation.setCalcName(name)

    calculation.load_molecule_fromFile(molecule_path)
    P = [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]
    calculation.makeSupercell(P)

    #  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #  n_gpus = torch.cuda.device_count()
    #  print("Number of cuda devices --> %s" % n_gpus,)
    #  model_path = os.path.join(MODEL_DIR, "best_model")
    #  model_schnet = load_model(model_path)

    if calc_type == "DFT":
        #  calculation.setQEspressoCalculatorV2(name, 20)
        calculation.setSiestaCalculator(name=file_base,
                                        dispCorrection="dftd4",
                                        nproc=56)
        calculation.setCalcName(name)
    elif calc_type == "model":
        from schnetpack.environment import AseEnvironmentProvider
        from schnetpack.utils import load_model
        import schnetpack

        import torch
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        # in case multiprocesses, global device variable rise CUDA spawn error.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #  n_gpus = torch.cuda.device_count()
        #  print("Number of cuda devices --> %s" % n_gpus,)
        #  device = "cuda"


        model_path = os.path.join(MODEL_DIR, "best_model")
        model_schnet = load_model(model_path, map_location=device)
        if "stress" in properties:
            print("Stress calculations are active")
            schnetpack.utils.activate_stress_computation(model_schnet)

        calculation.setSchnetCalcultor(
            model_schnet,
            properties,
            environment_provider=AseEnvironmentProvider(cutoff=5.5),
            device=device,
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

        calculation.optimize(fmax=0.02)
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


        traj_name = "%s_%s_EOS.traj" % (file_base, calc_type)
        traj = Trajectory(traj_name, "w")
        scaleFs = np.linspace(0.95, 1.10, 8)
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


def p_run(idx):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx % 2)

    mof_num = idx

    #  for bulk MOFs
    file_base = "IRMOF%s" %mof_num

    MOL_DIR = BASE_DIR + "/geom_files/IRMOFSeries/cif_files/"
    molecule_path = os.path.join(MOL_DIR, "%s.cif" %file_base)

    #  for fragments
    #  MOL_DIR = BASE_DIR + "/prepare_data/geomFiles/IRMOFSeries/fragments_6/"
    #  file_base = "irmofseries%s_f6" %mof_num
    #  molecule_path = os.path.join(MOL_DIR, "%s.xyz" %file_base)


    #  db_path = os.path.join(DB_DIR, "nonEquGeometriesEnergyForcesWithORCAFromMD.db")
    #data = AtomsData(path_to_db)
    #db_atoms = data.get_atoms(0)

    #  run(calc_type="DFT", run_type="opt")
    run(file_base, molecule_path, calc_type, run_type)


if __name__ == "__main__":
    from multiprocessing import Pool

    #  mof_nums = [10, 4, 6, 7, 1]
    mof_nums = [1, 4, 6, 7, 10]
    #  file_base = "mof5_mercury"
    #  mof_num = "1"
    calc_type = "DFT"
    #  calc_type = "model"
    run_type = "EOS"
    #  run_type = "opt"

    BASE_DIR = "/truba_scratch/otayfuroglu/deepMOF_dev/"
    MODEL_DIR = BASE_DIR \
            + "/schnetpack" \
            + "/works" \
            + "/runTrain" \
            + "/schnet_l3_basis96_filter64_interact3_gaussian20_rho001_lr00001_bs1_cutoff_60_"\
            + "withoutStress_aseEnv_test_IRMOFseries1_4_6_7_10_merged_173014_ev"
    #  + "/hdnnBehler_l3n100_rho001_r20a5_lr0001_bs1_IRMOFseries%s_merged_72886_ev" %mof_num\
    properties = ["energy", "forces", "stress"]  # properties used for training

    with Pool(1) as pool:
       pool.map(p_run, mof_nums)

#
from calculationsWithAse import AseCalculations
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import Trajectory
from ase.io.xyz import read_xyz,  write_xyz
from ase.io import read, write
from ase.optimize import BFGS, LBFGS, GPMin, QuasiNewton

import time
import numpy as np
import os, shutil
#  import tqdm

def getWorksDir(calc_name):

    WORKS_DIR = calc_name
    if os.path.exists(WORKS_DIR):
        shutil.rmtree(WORKS_DIR)
    os.mkdir(WORKS_DIR)

    return WORKS_DIR


def run(atoms, calc_type, run_type, temp=0, cell=1):

    name = "%s_%s_%sK" % (calc_type, run_type, temp)
    CW_DIR = os.getcwd()

    # main directory for caculation runOpt
    if not os.path.exists("ase_worksdir"):
        os.mkdir("ase_worksdir")

    WORKS_DIR = getWorksDir(RESULT_DIR + f"/ase_worksdir/{name}")

    os.chdir(WORKS_DIR)

    calculation = AseCalculations(WORKS_DIR)
    calculation.setCalcName(name)

    #  calculation.load_molecule_fromFile(molecule_path)
    calculation.load_molecule_fromAseatoms(atoms)
    P = [[0, 0, -cell], [0, -cell, 0], [-cell, 0, 0]]
    calculation.makeSupercell(P)

    if calc_type.lower() in ["schnetpack", "ani"]:
        import torch
        # in case multiprocesses, global device variable rise CUDA spawn error.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpus = torch.cuda.device_count()
        print("Number of cuda devices --> %s" % n_gpus,)


    # set calculator type
    if calc_type.lower() == "dft":
        #  calculation.setQEspressoCalculatorV2(name, 20)
        calculation.setSiestaCalculator(name=file_base,
                                        dispCorrection="dftd4",
                                        nproc=56)
        calculation.setCalcName(name)


    elif calc_type.lower() == "schnetpack":
        from schnetpack.environment import AseEnvironmentProvider
        from schnetpack.utils import load_model
        import schnetpack

        model_path = os.path.join(args.MODEL_DIR, "best_model")
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

    elif calc_type.lower() == "ani":
        calculation.setAniCalculator(model_type="ani2x", device=device)

    elif calc_type.lower() == "n2p2":
        calculation.setN2P2Calculator(
            args.MODEL_DIR,
            best_epoch=66,
            energy_units="eV",
            length_units="eV/Angstrom",
        )


    elif calc_type.lower() == "nequip":
        calculation.setNequipCalculator(args.MODEL_DIR, device="cuda")
    # set calculation type
    if run_type.lower() == "opt":
        calculation.optimize()
        os.chdir(CW_DIR)

    elif run_type.lower() == "vibration":
        #  WORKS_DIR = WORKS_DIR.replace("vibration", "opt").replace("DFT", "model")
        #  opt_molpath= "%s/Optimization.xyz" %WORKS_DIR
        #  calculation.load_molecule_fromFile(molecule_path=opt_molpath)
        calculation.optimize(fmax=0.001)
        calculation.vibration(nfree=2)
        os.chdir(CW_DIR)

    elif run_type.lower() == "md":

        calculation.init_md(
          name=name,
          time_step=0.5,
          temp_init=100.0,
          # temp_bath should be None for NVE and NPT
          temp_bath=temp,
          # temperature_K for NPT
          #  temperature_K=temp,
          interval=50,
        )

        calculation.optimize(fmax=0.0005)
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

    elif run_type.lower() == "eos":

        cell = calculation.molecule.get_cell()


        traj_name = "%s_%s_EOS.traj" % (file_base, calc_type)
        traj = Trajectory(traj_name, "w")
        scaleFs = np.linspace(0.95, 1.10, 8)
        print("Starting EOS calculations")
        print("Number of scaling factor values:", len(scaleFs))
        for scaleF in scaleFs:
            calculation.molecule.set_cell(cell * scaleF, scale_atoms=True)
            calculation.get_potential_energy()

            traj.write(calculation.molecule)
        os.chdir(CW_DIR)


    elif run_type.lower() == "optlattice":
        name = file_base + "_calc_lattice_const"
        from ase.constraints import StrainFilter, UnitCellFilter

        #  calculation.setCalcName("mof5Lattice")
        #  calculation.molecule.set_pbc([True, True, True])
        print(calculation.molecule.get_cell_lengths_and_angles())
        calculation.optimizeWithStrain(fmax=0.0005)
        print(calculation.molecule.get_cell_lengths_and_angles())

        calculation.optimize(fmax=0.05)


def p_run(idxs):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idxs % 2)

    #  db_path = os.path.join(DB_DIR, "nonEquGeometriesEnergyForcesWithORCAFromMD.db")
    #data = AtomsData(path_to_db)
    #db_atoms = data.get_atoms(0)


    #  molecule_path = os.path.join(MOL_DIR, file_name)
    atoms = atoms_list[idxs]
    run(atoms, calc_type, run_type, temp, cell)


if __name__ == "__main__":
    from multiprocessing import Pool
    import argparse

    parser = argparse.ArgumentParser(description="Give something ...")
    parser.add_argument("-run_type", "--run_type",
                        type=str, required=True,
                        help="..")
    parser.add_argument("-calc_type", "--calc_type",
                        type=str, required=True,
                        help="..")
    #  parser.add_argument("-temp_list", "--temp_list", nargs='+',
    #                     default=[], type=int)
    parser.add_argument("-temp", "--temp", type=int)
    parser.add_argument("-cell", "--cell",
                        type=int, required=True,
                        help="..")
    parser.add_argument("-MODEL_DIR", "--MODEL_DIR",
                        type=str, required=False,
                        help="..")
    parser.add_argument("-RESULT_DIR", "--RESULT_DIR",
                        type=str, required=True,
                        help="..")
    parser.add_argument("-mol_path", "--mol_path",
                        type=str, required=True,
                        help="..")
    args = parser.parse_args()

    run_type = args.run_type
    calc_type = args.calc_type
    temp = args.temp
    cell = args.cell
    RESULT_DIR = args.RESULT_DIR
    mol_path = args.mol_path
    properties = ["energy", "forces", "stress"]  # properties used for training


    atoms_list = read(mol_path, index=":")
    #  if calc_type.lower() == "nequip":
    if len(atoms_list) == 1:
        atoms = atoms_list[0]
        run(atoms, calc_type, run_type, temp, cell)

    else:
        n_atoms = len(atoms_list)
        idxs = range(n_atoms)
        nprocs = n_atoms
        with Pool(nprocs) as pool:
           pool.map(p_run, idxs)

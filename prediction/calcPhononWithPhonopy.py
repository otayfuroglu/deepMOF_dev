#! /truba/home/yzorlu/miniconda3/bin/python -u

from ase import Atoms
from ase.io import read
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.units import VaspToCm, SiestaToTHz, THzToCm, AseToCm
import numpy as np

from schnetpack.interfaces import SpkCalculator
from schnetpack.utils import load_model
from schnetpack import Properties
from schnetpack.datasets import AtomsData
from schnetpack.environment import AseEnvironmentProvider

import os


def get_crystalFromAseMol(atoms):
    cell = PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                        cell=atoms.cell[:], # for get 3x3 diognal marix of cell lengths
                        scaled_positions=atoms.get_scaled_positions())
    return cell

def optStructure(atoms, calculator):
    from ase.optimize import BFGS, LBFGS

    atoms.set_calculator(calculator)
    opt = LBFGS(atoms)
    opt.run(fmax=0.001)

    return atoms

def calc_siesta(nproc):
    from ase.units import Ry
    from ase.calculators.siesta import Siesta
    os.environ["ASE_SIESTA_COMMAND"] = "mpirun -np %s $SIESTA_PATH < PREFIX.fdf > PREFIX.out" %nproc
    if not os.path.exists("work_siesta"):
        os.mkdir("work_siesta")

    label = "work_siesta/siesta"
    calculator = Siesta(
        label=label,
        xc="PBE",
        mesh_cutoff=700 * Ry,
        energy_shift=0.01 * Ry,
        basis_set='TZP',
        kpts=[2, 2, 2],
        fdf_arguments={"DM.MixingWeight": 0.25,
                       "DM.NumberPulay" : 1,
                       "MaxSCFIterations": 100,
                      },
    )

    return calculator


def phonopy_pre_process(cell, supercell_matrix=None):

    if supercell_matrix is None:
        smat = [[2,0,0], [0,2,0], [0,0,2]],
    else:
        smat = supercell_matrix
    phonon = Phonopy(cell,
                     smat,
                     primitive_matrix=[[0.0, 0.5, 0.5],
                                       [0.5, 0.0, 0.5],
                                       [0.5, 0.5, 0.0]],
                    factor=AseToCm,
                     #  factor=SiestaToTHz * THzToCm
                    )
    phonon.generate_displacements(distance=0.03)
    print("[Phonopy] Atomic displacements:")
    disps = phonon.get_displacements()
    for d in disps:
        print("[Phonopy] %d %s" % (d[0], d[1:]))
    return phonon


def run_calc(calc, phonon):
    supercells = phonon.get_supercells_with_displacements()
    # Force calculations by calculator
    set_of_forces = []
    for scell in supercells:
        print("SUPERCELL ", scell)
        cell = Atoms(symbols=scell.get_chemical_symbols(),
                     scaled_positions=scell.get_scaled_positions(),
                     cell=scell.get_cell(),
                     pbc=True)
        cell.set_calculator(calc)
        forces = cell.get_forces()
        drift_force = forces.sum(axis=0)
        print(("[Phonopy] Drift force:" + "%11.5f" * 3) % tuple(drift_force))
        # Simple translational invariance
        for force in forces:
            force -= drift_force / forces.shape[0]
        set_of_forces.append(forces)
    return set_of_forces

def phonopy_post_process(phonon, set_of_forces):
    phonon.produce_force_constants(forces=set_of_forces)
    print('')
    print("[Phonopy] Phonon frequencies at Gamma:")
    for i, freq in enumerate(phonon.get_frequencies((0, 0, 0))):
        print("[Phonopy] %3d: %10.5f THz" %  (i + 1, freq)) # THz

    # DOS 
    # projected
    #  phonon.run_mesh([10, 10, 10], with_eigenvectors=True, is_mesh_symmetry=False)
    #  phonon.set_total_DOS(tetrahedron_method=True)
    #  phonon.run_projected_dos()
    #  phonon.plot_projected_dos().savefig("dos.png")

    #total
    phonon.run_mesh([32, 32, 32])
    phonon.run_total_dos()
    phonon.plot_total_dos().savefig("dft_tatal_dos.png")
    phonon.write_total_dos()

    print('')
    print("[Phonopy] Frequency Total DOS:")
    for omega, dos in np.array(phonon.get_total_DOS()).T:
        print("%15.7f%15.7f" % (omega, dos))

def main():
    device="cuda"
    BASE_DIR = "/truba_scratch/yzorlu/deepMOF/HDNNP"
    MODEL_DIR = "schnet_l3_basis96_filter64_interact3_gaussian20_rho001_lr00001_bs1_cutoff_60_withoutStress_aseEnv_IRMOFseries1_4_6_7_10_merged_173014_ev"
    model_schnet = load_model("%s/schnetpack/runTraining/%s/best_model" %(BASE_DIR, MODEL_DIR))
    calc_schnet = SpkCalculator(model_schnet, device=device,
                                energy=Properties.energy,
                                forces=Properties.forces,
                                #  collect_triples=True,
                                environment_provider=AseEnvironmentProvider(cutoff=6.0))

    file_name = "mercury_IRMOF1.cif"
    atoms = read("%s/prepare_data/geomFiles/IRMOFSeries/cif_files/%s" %(BASE_DIR, file_name))
    #  calc = calc_schnet
    calc = calc_siesta(40)
    optStructure(atoms, calculator=calc_schnet)

    # 1x1x1 supercell of conventional unit cell
    cell = get_crystalFromAseMol(atoms)

    phonon = phonopy_pre_process(cell, supercell_matrix=np.eye(3, dtype='intc'))

    # # 2x2x2 supercell of conventional unit cell
    # calc = get_gpaw(kpts_size=(2, 2, 2))
    # phonon = phonopy_pre_process(cell,
    #                              supercell_matrix=(np.eye(3, dtype='intc') * 2))

    set_of_forces = run_calc(calc, phonon)
    phonopy_post_process(phonon, set_of_forces)

if __name__ == "__main__":
    main()

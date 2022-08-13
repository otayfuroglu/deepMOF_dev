#
"""
!!! This module was created by quoting from Schnetpack!!!

This module provides a ASE calculator class [#ase1]_ for SchNetPack models, as
well as a general Interface to all ASE calculation methods, such as geometry
optimisation, normal mode computation and molecular dynamics simulations.
References
----------
.. [#ase1] Larsen, Mortensen, Blomqvist, Castelli, Christensen, Dułak, Friis,
    Groves, Hammer, Hargus: The atomic simulation environment -- a Python
        library for working with atoms.
            Journal of Physics: Condensed Matter, 9, 27. 2017.

"""

from ase.io import read
from ase.optimize import BFGS, LBFGS, GPMin, QuasiNewton
from ase.io.trajectory import Trajectory
from ase.io.xyz import read_xyz, write_xyz
from ase.io import write
from ase.constraints import StrainFilter, UnitCellFilter

from ase.md import VelocityVerlet, Langevin, MDLogger
from ase.md.nptberendsen import NPTBerendsen, Inhomogeneous_NPTBerendsen
from ase.md.npt import NPT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase import units
from ase.calculators.dftd3 import DFTD3
#  from ase.calculators.mixing import SumCalculator
#  from dftd4.ase import DFTD4

import numpy as np
import os
import tqdm

class AseCalculations(object):
    """
    x
    """

    def __init__(self, working_dir):

        self.working_dir = working_dir
        self.calcName = None

        #load molecule
        self.molecule = None
        #self._load_molecule(molecule_path, db_atoms)

        # unless initialized, set dynamics to False
        self.dynamics = None

    def setQEspressoCalculator(self, name, nproc):

        #  from ase.calculators.espresso import Espresso
        from qeutil import QuantumEspresso

        calculator = QuantumEspresso(
            label=name,                    # Label for calculations
            wdir="runEspresso",                   # Working directory
            pseudo_dir="/truba/home/yzorlu/pseudo_pot/qe/all_lda_UPF_v1.5",   # Directory with pseudopotentials
            kpts=[2,2,2],   # K-space sampling for the SCF calculation
            xc='pz',        # Exchange functional type in the name of the pseudopotentials
            pp_type='vbc',  # Variant of the pseudopotential
            pp_format='UPF',# Format of the pseudopotential files
            ecutwfc=15,     # Energy cut-off (in Rydberg)
            occupations='smearing',
            smearing='methfessel-paxton',
            degauss=0.0198529,
            mixing_beta=0.7, mixing_mode = 'plain', diagonalization = 'cg',
            restart_mode='from_scratch', tstress=False,
            use_symmetry=True, # Use symmetry in the calculation
            procs=nproc,
        )

        self.molecule.set_calculator(calculator)

    def setQEspressoCalculatorV2(self, name, nproc):
        from ase.calculators.espresso import Espresso

        #  os.system(". ./setCalculatorsEnvrionmet.sh")
        os.environ["ASE_ESPRESSO_COMMAND"] = "mpirun -np %s $ESPRESSO_DIR/bin/pw.x -in PREFIX.pwi > PREFIX.pwo" %nproc
        pseudopotentials = {
            'H': 'h_lda_v1.4.uspp.F.UPF',
            'C': 'c_lda_v1.2.uspp.F.UPF',
            'O': 'o_lda_v1.2.uspp.F.UPF',
            'Zn': 'zn_lda_v1.uspp.F.UPF',
                           }

        calculator = Espresso(
            label=name,
            pseudopotentials=pseudopotentials,
            #controls
            verbosity="high",
            restart_mode="from_scratch",
            wf_collect=False,
            tstress=True,
            wfcdir=self.working_dir,
            etot_conv_thr=1.0E-4,
            forc_conv_thr=1.0E-3,
            disk_io="low",
            #system
            input_dft="PBE",
            vdw_corr="DFT-D3",
            ecutwfc=64,
            ecutrho=576,
            #electrons
            electron_maxstep=100,
            scf_must_converge=True,
            conv_thr=1.0E-6,

            #  input_data=input_data,
        )

        self.molecule.set_calculator(calculator)

    def setSiestaCalculator(self, name, dispCorrection, nproc):
        from ase.units import Ry
        from ase.calculators.siesta import Siesta
        os.environ["ASE_SIESTA_COMMAND"] = "mpirun -np %s $SIESTA_PATH < PREFIX.fdf > PREFIX.out" %nproc

        dft_calculator = Siesta(
            label=name,
            xc="PBE",
            mesh_cutoff=300 * Ry,
            energy_shift=0.01 * Ry,
            basis_set='DZP',
            kpts=[4, 4, 4],
            fdf_arguments={"DM.MixingWeight": 0.25,
                           "DM.NumberPulay" : 1,
                           "MaxSCFIterations": 100,
                          },
        )

        #for D3 correction
        print("Used Grimm's %s dispersion correction. " %dispCorrection)
        if dispCorrection.lower() == "dftd3":
            calculator = DFTD3(xc="pbe", dft=dft_calculator)

        elif dispCorrection.lower() == "dftd4":
            from ase.calculators.mixing import SumCalculator
            from dftd4.ase import DFTD4

            calculator = SumCalculator([DFTD4(method="PBE"), dft_calculator])
        else:
            calculator = dft_calculator
        self.molecule.set_calculator(calculator)

    def setOrcaCalculator(self, label, n_cpu, initial_gbw=["", ""]):
        from ase.calculators.orca import ORCA
        self._moleculeCheck()

        calculator = ORCA(
            label=label,
            maxiter=200,
            charge=0, mult=1,
            orcasimpleinput='SP PBE D4 DEF2-TZVP DEF2/J RIJDX MINIPRINT NOPRINTMOS NOPOP NoKeepInts NOKEEPDENS '\
                            + ' ' + initial_gbw[0],
            orcablocks= '%scf Convergence normal \n maxiter 40 end \n %pal nprocs ' + str(n_cpu) + ' end' + initial_gbw[1])
        self.molecule.set_calculator(calculator)

    def setSchnetCalcultor(self, model, properties, environment_provider, device):
        from schnetpack.interfaces import SpkCalculator
        self._moleculeCheck()

        try:
            stress = properties[2]
        except:
            stress = None

        calculator = SpkCalculator(
            model,
            device=device,
            collect_triples=False,
            environment_provider=environment_provider,
            energy=properties[0],
            forces=properties[1],
            stress=stress,
            energy_units="eV",
            forces_units="eV/Angstrom",
            stress_units="eV/Angstrom/Angstrom/Angstrom",
        )

        self.molecule.set_calculator(calculator)

    def setAniCalculator(self, model_type="ani2x", device="cuda", dispCorrection=None):
        import torchani
        if "ani1x" == model_type:
            ani_calculator = torchani.models.ANI1x().to(device).ase()
        elif "ani1ccx" == model_type:
            ani_calculator = torchani.models.ANI1ccx().to(device).ase()
        elif "ani2x" == model_type:
            ani_calculator = torchani.models.ANI2x().to(device).ase()


        #for D3 correction
        print("Used Grimm's %s dispersion correction. " %dispCorrection)
        if dispCorrection is None:
            calculator = ani_calculator
        elif dispCorrection.lower() == "dftd3":
            calculator = DFTD3(xc="pbe", dft=ani_calculator)

        elif dispCorrection.lower() == "dftd4":
            from ase.calculators.mixing import SumCalculator
            from dftd4.ase import DFTD4

            calculator = SumCalculator([DFTD4(method="PBE"), ani_calculator])
        self.molecule.set_calculator(calculator)


    def setN2P2Calculator(self, model_dir, best_epoch):
        import sys
        sys.path.insert(1, "/truba_scratch/yzorlu/deepMOF_dev") #TODO makeshift
        from  n2p2.prediction.n2p2AseInterFace import n2p2Calculator

        calculator = n2p2Calculator(model_dir, best_epoch)
        self.molecule.set_calculator(calculator)


    def setQMMMcalculator(self, qm_region, qm_calcultor, mm_calcultor):
        from ase.calculators.qmmm import SimpleQMMM
        self.molecule.set_calculator(SimpleQMMM(
            qm_region, qm_calcultor, mm_calcultor, mm_calcultor
        ))

    def setQMMMForceCalculator(self, qm_selection_mask, qm_calcultor, mm_calcultor, buffer_width):
        from ase.calculators.qmmm import ForceQMMM, RescaledCalculator

        qmmm_calc = ForceQMMM(self.molecule, qm_selection_mask,
                              qm_calcultor, mm_calcultor, buffer_width,
                              vacuum=0.0,
                             )
        self.molecule.set_calculator(qmmm_calc)

    def setEIQMMMCalculator(self, qm_selection, qm_calcultor, mm_calcultor):
        from ase.calculators.qmmm import (EIQMMM, Embedding,
                                          LJInteractions,
                                          LJInteractionsGeneral)
        from ase.calculators.tip3p import epsilon0, sigma0

        # General LJ interaction object for the 'OHHOHH' water dimer
        #  sigma_mm = np.array([sigma0, 0, 0])  # Hydrogens have 0 LJ parameters
        #  epsilon_mm = np.array([epsilon0, 0, 0])
        #  sigma_qm = np.array([sigma0, 0, 0])
        #  epsilon_qm = np.array([epsilon0, 0, 0])

        qm_molecule_size = len(qm_selection)
        mm_molecule_size = len(self.molecule) - qm_molecule_size
        #  interaction = LJInteractionsGeneral(sigma_qm, epsilon_qm,
        #                                      sigma_mm, epsilon_mm,
        #                                      qm_molecule_size,
        #                                      mm_molecule_size)
        interaction = LJInteractions({('O', 'O'): (epsilon0, sigma0)})

        embedding=Embedding(mm_molecule_size)

        qmmm_calc = EIQMMM(
            qm_selection, qm_calcultor, mm_calcultor,
            interaction=interaction, embedding=embedding,
            vacuum=None
        )
        self.molecule.set_calculator(qmmm_calc)

    def attach_molecule(self, molecule, n_molecule, distance):
        from ase.build.attach import attach_randomly
        for _ in range(n_molecule):
            self.molecule = attach_randomly(self.molecule, molecule, distance=3.0)

    def _moleculeCheck(self):
        if self.molecule is None:
            raise AttributeError(
                "Molecule need to be loading using the load_molecule_from* function"
            )

    def setCalcName(self, calcName):
        self.calcName = calcName

    def load_molecule_fromFile(self, molecule_path):
        self.molecule = read(molecule_path)

    def load_molecule_fromDB(self, db_atoms):
        self.molecule = db_atoms

    def load_molecule_fromAseatoms(self, ase_atoms):
        self.molecule = ase_atoms

    def makeSupercell(self, P):
        from ase.build import make_supercell
        self.molecule = make_supercell(self.molecule, P)

    def save_molecule(self, write_format="xyz"):

        molecule_path = os.path.join(self.working_dir, "%s.%s" % (self.calcName, write_format))
        if write_format == "ext_xyz":
            write_xyz(molecule_path, self.molecule, plain=False)
        else:
            write(molecule_path, self.molecule)

    def get_potential_energy(self):
        return self.molecule.get_potential_energy()

    def calculate_single_point(self):
        self.molecule.energy = self.molecule.get_potential_energy()
        self.molecule.forces = self.molecule.get_forces()
        #self.save_molecule("single_point", write_format="extxyz")

    def optimize(self, fmax=1.0e-2, steps=1000):
        self.calcName = "Optimization"

        optimization_path = os.path.join(self.working_dir, self.calcName)
        optimizer = LBFGS(self.molecule,
                                #  trajectory="%s.traj" % optimization_path,
                                #  restart="%s.pkl" % optimization_path,
                               )
        optimizer.run(fmax, steps)
        self.save_molecule()

    def vibration(self, delta=0.01, nfree=2):
        from ase.vibrations import Vibrations

        vib = Vibrations(self.molecule,
                         name=self.calcName,
                         delta=delta, nfree=nfree)
        vib.run()
        #  vib.summary()
        vib.write_jmol()
        vib.write_dos(out="vib-dos_Lorentzian_5_width.dat", start=0, end=4000, type="Lorentzian", width=5)
        vib.write_dos(out="vib-dos_Lorentzian_10_width.dat", start=0, end=4000, type="Lorentzian", width=10)
        vib.write_dos(out="vib-dos_Lorentzian_20_width.dat", start=0, end=4000, type="Lorentzian", width=20)
        vib.write_dos(out="vib-dos_Lorentzian_30_width.dat", start=0, end=4000, type="Lorentzian", width=30)
        vib.write_dos(out="vib-dos_Gaussian_5_width.dat", start=0, end=4000, type="Gaussian", width=5)
        vib.write_dos(out="vib-dos_Gaussian_10_width.dat", start=0, end=4000, type="Gaussian", width=10)
        vib.write_dos(out="vib-dos_Gaussian_20_width.dat", start=0, end=4000, type="Gaussian", width=20)
        vib.write_dos(out="vib-dos_Gaussian_30_width.dat", start=0, end=4000, type="Gaussian", width=30)
        vib.clean()

    def optimizeWithStrain(self, fmax=1.0e-1, steps=1000):
        self.calcName = "Optimization"

        optimization_path = os.path.join(self.working_dir, self.calcName)
        molecule = StrainFilter(self.molecule)
        optimizer = LBFGS(molecule,
                         trajectory="%s.traj" % optimization_path,
                        )
        optimizer.run(fmax, steps)

    def init_md(
        self,
        name,
        time_step=0.2,
        temp_init=100,
        temp_bath=None,
        temperature_K=None,
        pressure=1, #bar
        reset=False,
        interval=1,
    ):

        # If a previous dynamics run has been performed, don't reinitialize
        # velocities unless explicitly requested via restart=True
        if not self.dynamics or reset:
            self._init_velocities(temp_init=temp_init)

        # setup dynamics
        if temperature_K:
            print("NPT ensemble.. Pressure set to %s bar" %pressure)
            #  self.dynamics = NPTBerendsen(
            #      self.molecule,
            #      timestep=time_step * units.fs,
            #      temperature_K=temperature_K,
            #      taut=100 * units.fs,
            #      pressure_au=pressure * 1.01325 * units.bar,
            #      taup=1000 * units.fs,
            #      compressibility=4.57e-5 / units.bar)

            externalstress = 1.0*units.Pascal  # (-1.7, -1.7, -1.7, 0, 0, 0)*GPa
            ttime = 100*units.fs
            ptime = 300*units.fs
            #  bulk_modulus = 35*units.GPa
            #  pfactor = (ptime**2)*bulk_modulus

            self.dynamics = NPT(self.molecule,
                                5*units.fs,
                                temperature_K=temperature_K,
                                externalstress=externalstress,
                                ttime=ttime,
                                #  pfactor=pfactor
                               )

        elif temp_bath:
            self.dynamics = Langevin(
                self.molecule,
                time_step * units.fs,
                temp_bath * units.kB,
                1.0 / (100.0 * units.fs),
            )
        else:
            self.dynamics = VelocityVerlet(self.molecule, time_step * units.fs)

        # Create monitors for logfile and traj file
        logfile = os.path.join(self.working_dir, "%s.log" % name)
        trajfile = os.path.join(self.working_dir, "%s.traj" % name)
        logger = MDLogger(
            self.dynamics,
            self.molecule,
            logfile,
            stress=False,
            peratom=False,
            header=True,
            mode="a",
        )
        trajectory = Trajectory(trajfile, "w", self.molecule)

        # Attach motiors to trajectory
        self.dynamics.attach(self.printMD)
        self.dynamics.attach(self.tqdmMD)
        self.dynamics.attach(logger, interval=interval)
        self.dynamics.attach(trajectory.write, interval=interval)

    def _init_velocities(
        self,
        temp_init=100,
        remove_translation=True,
        remove_rotation=True,
    ):
        """
        Initialize velocities for MD
        """
        MaxwellBoltzmannDistribution(self.molecule, temp_init * units.kB)
        if remove_translation:
            Stationary(self.molecule)
            if remove_rotation:
                ZeroRotation(self.molecule)

    def _createOutFile(self, file_path):
        self.out_file = open(file_path, "w")
        print('Epot,Temperature,Volume', file=self.out_file)

    def printMD(self):
        """xxx"""
        volume = self.molecule.get_volume()
        temp = self.molecule.get_temperature()
        epot = self.molecule.get_potential_energy()

        print('%.4f,%.4f,%.4f'%(epot, temp, volume), file=self.out_file)

    def _setTqdm(self, steps):
        #  self.pbar =
        self.pbar = tqdm.tqdm(total=steps)

    def tqdmMD(self):
        self.pbar.update()

    def run_md(self, steps):
        if not self.dynamics:
            raise AttributeError(
                "Dynamics need to be initialize using the setup_md function"
            )
        self._setTqdm(steps)
        out_file_path = os.path.join(self.working_dir, "%s.out" % self.calcName)
        self._createOutFile(out_file_path)
        self.dynamics.run(steps)

    def print_calc(self):
        print(self.molecule.get_potential_energy())



import numpy as np

from ase import Atoms
from ase.io import read, write
from molmod.units import *
from molmod.constants import *
from molmod.periodic import periodic
from utilities import _random_rotation, random_position, vdw_overlap
import tqdm

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs, kB
from ase.io.trajectory import Trajectory

from itertools import combinations



class AI_GCMCMD():
    def __init__(self, model, results_dir, interval, atoms_frame, atoms_ads, flex_ads, T, P, fugacity, device, vdw_radii):
        self.model = model
        self.results_dir = results_dir
        self.interval = interval
        self.atoms_frame = atoms_frame
        self.n_frame = len(self.atoms_frame)
        self.atoms_ads = atoms_ads
        self.flex_ads = flex_ads
        self.n_ads = len(self.atoms_ads)
        self.cell = np.array(self.atoms_frame.get_cell())
        self.V = np.linalg.det(self.cell) * angstrom**3
        self.T = T
        self.P = P
        self.fugacity = fugacity
        self.device = device
        self.beta = 1 / (boltzmann * T)
        self.Z_ads = 0

        self.vdw = vdw_radii - 0.35

    def _insertion_acceptance(self, e_trial, e):
        exp_value = self.beta * (e - e_trial)
        if exp_value > 100:
            return True
        elif exp_value < -100:
            return False
        else:
            acc = min(1, self.V*self.beta*self.fugacity/self.Z_ads * np.exp(exp_value))
            return np.random.rand() < acc

    def _deletion_acceptance(self, e_trial, e):
        exp_value = -self.beta * (e_trial - e)
        if exp_value > 100:
            return True
        else:
            acc = min(1, (self.Z_ads+1)/self.V/self.beta/self.fugacity * np.exp(exp_value))
            return np.random.rand() < acc

    def _get_ads_atoms(self, atoms):

        ads_atoms = atoms[self.n_frame:]

        ads_molecules = []
        for i in range(0, len(ads_atoms), self.n_ads):
            ads_molecule = Atoms(
                symbols = ads_atoms[i:i+self.n_ads].get_chemical_symbols(),
                positions = ads_atoms[i:i+self.n_ads].get_positions()
            )
            ads_molecules.append(ads_molecule)

        # Combine all CO₂ molecules into a single Atoms object
        #  ads_system = sum(ads_molecules, start=Atoms())
        #  write("test_ads_molecules.extxyz", ads_system)
        assert len(ads_molecules) == self.Z_ads

        return ads_molecules

    def _set_rigid_ads_atoms(self, atoms):

        from ase.constraints import FixBondLengths

        def split_list(lst, length):
            """Split a list into sublists of the specified length."""
            return [lst[i:i + length] for i in range(0, len(lst), length)]

        indices = [atom.index for atom in atoms]
        ads_indices = indices[self.n_frame:]

        for ad_indices in split_list(ads_indices, self.n_ads):
            fix_bond_indices = list(combinations(ad_indices, 2))[:-1]
            c = FixBondLengths(fix_bond_indices)
            atoms.set_constraint(c)

    def _get_e_ads(self, atoms):
        e_ads = 0.0
        for ads_atoms in self._get_ads_atoms(atoms):
            ads_atoms.calc = self.model
            e_ads += ads_atoms.get_potential_energy()
        return e_ads

    def _run_nvt_md(self, atoms, timestep, N):
        from ase.md.langevin import Langevin

        if not self.flex_ads:
            self._set_rigid_ads_atoms(atoms)

        atoms.calc = self.model
        # Set initial velocities corresponding to T
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.T)
        trajectory = Trajectory(f'{self.results_dir}/trajectory_{self.P/bar}bar.traj', "a", atoms)
        # Define the Langevin dynamics
        dyn = Langevin(atoms,
                       timestep=timestep*fs,        # Timestep of 1 femtosecond
                       temperature_K=self.T,      # Target temperature in Kelvin
                       friction=0.01)          # Friction coefficient
        dyn.attach(trajectory.write, interval=self.interval)  # Write every step
        #  self._setTqdm(N)
        #  dyn.attach(self._tqdmMD)
        # Run the dynamics for a fixed number of steps
        dyn.run(N)

        # to remove contraints after MD
        if not self.flex_ads:
            del atoms.constraints

    def _run_npt_md(self, atoms, timestep, N):
        from ase.md.npt import NPT

        if not self.flex_ads:
            self._set_rigid_ads_atoms(atoms)

        atoms.calc = self.model
        # Set initial velocities corresponding to T
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.T)
        trajectory = Trajectory(f'{self.results_dir}/trajectory_{self.P/bar}bar.traj', "a", atoms)
        # Define the NPT dynamics
        dyn = NPT(atoms,
                  timestep=timestep*fs,  # Timestep of 1 femtosecond
                  temperature_K=self.T,
                  externalstress=self.P,
                  ttime=25*fs,  # Thermostat coupling time
                  pfactor=0.6,)    # Barostat coupling factor

        dyn.attach(trajectory.write, interval=self.interval)  # Write every step
        #  self._setTqdm(N)
        #  dyn.attach(self._tqdmMD)
        # Run the dynamics for a fixed number of steps
        dyn.run(N)

        # to remove contraints after MD
        if not self.flex_ads:
            del atoms.constraints

    def _run_nptberendsen_md(self, atoms, timestep, N, Inhomogeneous=False):
        from ase.md.nptberendsen import NPTBerendsen, Inhomogeneous_NPTBerendsen

        if not self.flex_ads:
            self._set_rigid_ads_atoms(atoms)

        atoms.calc = self.model
        # Set initial velocities corresponding to T
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.T)
        trajectory = Trajectory(f'{self.results_dir}/trajectory_{self.P/bar}bar.traj', "a", atoms)
        # Define the NPT dynamics
        if not Inhomogeneous:
            dyn = NPTBerendsen(atoms,
                               timestep=timestep*fs,  # Timestep of 1 femtosecond
                               temperature_K=self.T,
                               pressure=self.P,
                               #  taut=0.5e3 *fs,
                               taut=1e2 *fs,
                               taup=1e3*fs,
                               compressibility=1e-7,
                              )
        else:
            dyn = Inhomogeneous_NPTBerendsen(atoms,
                               timestep=timestep*fs,  # Timestep of 1 femtosecond
                               temperature_K=self.T,
                               pressure=self.P,
                               #  taut=0.5e3 *fs,
                               taut=1e2 *fs,
                               taup=1e3*fs,
                               compressibility=1e-7,
                              )

        dyn.attach(trajectory.write, interval=self.interval)  # Write every step
        #  self._setTqdm(N)
        #  dyn.attach(self._tqdmMD)
        # Run the dynamics for a fixed number of steps
        dyn.run(N)

        # to remove contraints after MD
        if not self.flex_ads:
            del atoms.constraints

    def _setTqdm(self, steps):
        #  self.pbar =
        self.pbar = tqdm.tqdm(total=steps)

    def _tqdmMD(self):
        self.pbar.update()

    def run(self, timestep, totalsteps, N, X):

        atoms = self.atoms_frame.copy()

        if not self.flex_ads:
            atoms.calc = self.model

            atoms_ads = self.atoms_ads.copy()
            atoms_ads_e = self.atoms_ads.copy()
            atoms_ads_e.calc = self.model
            e_ads = atoms_ads_e.get_potential_energy() * electronvolt # eV to Hartree
        #  e = atoms.get_potential_energy() * electronvolt # eV to Hartree

        uptake = []
        adsorption_energy = []

        fl_status = open(f"{self.results_dir}/status.csv", "w")
        print("Steps,trial_insertion, succ_insertion,trial_deletaion,succ_deletion", file=fl_status)
        fl_status.flush()

        n_trial_insertion = 0
        n_succ_insertion = 0
        n_trial_deletion = 0
        n_succ_deletion = 0
        steps = 0
        old_steps = 0

        for iteration in tqdm.trange(totalsteps):
            if iteration % X == 0:
                #run md
                #  self._run_nvt_md(atoms, timestep, N)
                #  self._run_npt_md(atoms, timestep, N)
                self._run_nptberendsen_md(atoms, timestep, N, Inhomogeneous=False)

                #  if iteration == 0:
                #get potential energy after MD steps
                if self.flex_ads:
                    e = atoms.get_potential_energy() * electronvolt - self._get_e_ads(atoms) * electronvolt
                else:
                    e = atoms.get_potential_energy() * electronvolt - e_ads * self.Z_ads # eV to Hartree

                # set V after MD steps
                self.cell = np.array(atoms.get_cell())
                self.V = np.linalg.det(self.cell) * angstrom**3

            switch = np.random.rand()

            # insertion
            if switch < 0.25:
                n_trial_insertion += 1

                self.Z_ads += 1
                atoms_trial = atoms + self.atoms_ads.copy()
                pos = atoms_trial.get_positions()
                pos[-self.n_ads:] = random_position(pos[-self.n_ads:], atoms_trial.get_cell())
                atoms_trial.set_positions(pos)
                if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, self.Z_ads-1):
                    e_trial = 10**10 * kjmol
                else:
                    atoms_trial.calc = self.model
                    if self.flex_ads:
                        e_trial = atoms_trial.get_potential_energy() * electronvolt - self._get_e_ads(atoms_trial) * electronvolt
                    else:
                        e_trial = atoms_trial.get_potential_energy() * electronvolt - e_ads * self.Z_ads
                if self._insertion_acceptance(e_trial, e):
                    n_succ_insertion += 1
                    steps += 1

                    atoms = atoms_trial.copy()
                    e = e_trial
                else:
                    self.Z_ads -= 1

            # Deletion
            elif switch < 0.5:
                if self.Z_ads != 0:
                    n_trial_deletion += 1

                    i_ads = np.random.randint(self.Z_ads)
                    atoms_trial = atoms.copy()
                    self.Z_ads -= 1
                    del atoms_trial[self.n_frame + self.n_ads*i_ads : self.n_frame + self.n_ads*(i_ads+1)]
                    atoms_trial.calc = self.model
                    if self.flex_ads:
                        e_trial = atoms_trial.get_potential_energy() * electronvolt - self._get_e_ads(atoms_trial) * electronvolt
                    else:
                        e_trial = atoms_trial.get_potential_energy() * electronvolt - e_ads * self.Z_ads
                    if self._deletion_acceptance(e_trial, e):
                        n_succ_deletion += 1
                        e = e_trial

                        atoms = atoms_trial.copy()
                        steps += 1
                    else:
                        self.Z_ads += 1

            # Translation
            elif switch < 0.75:
                if self.Z_ads != 0:
                    i_ads = np.random.randint(self.Z_ads)
                    atoms_trial = atoms.copy()
                    pos = atoms_trial.get_positions()
                    pos[self.n_frame + self.n_ads*i_ads : self.n_frame + self.n_ads*(i_ads+1)] += 0.5 * (np.random.rand(3) - 0.5)
                    atoms_trial.set_positions(pos)
                    if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, i_ads):
                        e_trial = 10**10 * kjmol
                    else:
                        atoms_trial.calc = self.model
                        if self.flex_ads:
                            e_trial = atoms_trial.get_potential_energy() * electronvolt - self._get_e_ads(atoms_trial) * electronvolt
                        else:
                            e_trial = atoms_trial.get_potential_energy() * electronvolt - e_ads * self.Z_ads
                    acc = min(1, np.exp(-self.beta*(e_trial-e)))
                    if acc > np.random.rand():
                        atoms = atoms_trial.copy()
                        e = e_trial
                        steps += 1

            # Rotation
            elif switch > 0.75:
                if self.Z_ads != 0:
                    i_ads = np.random.randint(self.Z_ads)
                    atoms_trial = atoms.copy()
                    pos = atoms_trial.get_positions()
                    pos[self.n_frame + self.n_ads*i_ads : self.n_frame + self.n_ads*(i_ads+1)] = _random_rotation(pos[self.n_frame + self.n_ads*i_ads : self.n_frame + self.n_ads*(i_ads+1)], circlefrac = 0.1)
                    atoms_trial.set_positions(pos)
                    if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, i_ads):
                        e_trial = 10**10 * kjmol
                    else:
                        atoms_trial.calc = self.model
                        if self.flex_ads:
                            e_trial = atoms_trial.get_potential_energy() * electronvolt - self._get_e_ads(atoms_trial) * electronvolt
                        else:
                            e_trial = atoms_trial.get_potential_energy() * electronvolt - e_ads * self.Z_ads
                    acc = min(1, np.exp(-self.beta*(e_trial-e)))
                    if acc > np.random.rand():
                        atoms = atoms_trial.copy()
                        e = e_trial
                        steps += 1

            uptake.append(self.Z_ads)
            adsorption_energy.append(e)

            if steps % self.interval == 0 and old_steps != steps:
                write(f'{self.results_dir}/trajectory_{self.P/bar}bar.extxyz', atoms, append=True)
                print(f"{steps},{n_trial_insertion},{n_succ_insertion},{n_trial_deletion},{n_succ_deletion}", file=fl_status)

                np.save(f'{self.results_dir}/uptake_{self.P/bar}bar.npy', np.array(uptake))
                np.save(f'{self.results_dir}/adsorption_energy_{self.P/bar}bar.npy', np.array(adsorption_energy))

                fl_status.flush()
                old_steps = steps

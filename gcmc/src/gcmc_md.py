import numpy as np
import random

from ase import Atoms
from ase.io import read, write
from molmod.units import *
from molmod.constants import *
from molmod.periodic import periodic
from utilities import _random_rotation, random_position, vdw_overlap
import tqdm

from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen, Inhomogeneous_NPTBerendsen

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs, kB
from ase.io.trajectory import Trajectory

from itertools import combinations



class GCMCMD():
    def __init__(self, calc_gcmc, calc_md, results_dir, interval, atoms_frame, atoms_ads, flex_ads, T, P, fugacity, device, vdw_radii):
        self.calc_gcmc = calc_gcmc
        self.calc_md = calc_md
        self.results_dir = results_dir
        self.interval = interval
        self.atoms = atoms_frame
        self.n_frame = len(atoms_frame)
        self.atoms_ads = atoms_ads
        self.flex_ads = flex_ads
        self.n_ads = len(self.atoms_ads)
        self.cell = np.array(atoms_frame.get_cell())
        self.V = np.linalg.det(self.cell) * angstrom**3
        self.T = T
        self.P = P
        self.fugacity = fugacity
        self.device = device
        self.beta = 1 / (boltzmann * T)
        self.Z_ads = 0

        self.vdw = vdw_radii - 0.35
        self.dyn = None

    def _get_ads_atoms(selfs):

        ads_atoms = self.atoms[self.n_frame:]

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

    def _set_rigid_ads_atoms(self):

        from ase.constraints import FixBondLengths

        indices = [atom.index for atom in self.atoms]
        ads_indices = indices[self.n_frame:]

        assert len(ads_indices)/3 == self.Z_ads

        for ad_indices in self._split_ads_indices(ads_indices, self.n_ads):
            fix_bond_indices = list(combinations(ad_indices, 2))[:-1]
            c = FixBondLengths(fix_bond_indices)
            self.atoms.set_constraint(c)

    def _set_rigid_triatomic_ads_atoms(self):

        from ase.constraints import FixLinearTriatomic

        indices = [atom.index for atom in self.atoms]
        ads_indices = indices[self.n_frame:]

        assert len(ads_indices)/3 == self.Z_ads

        if self.Z_ads != 0:
            sub_ads_indices = self._split_ads_indices(ads_indices, self.n_ads)
            c = FixLinearTriatomic(triples=sub_ads_indices)
            self.atoms.set_constraint(c)

    def _split_ads_indices(self, lst, length):
        """Split a list into sub tuple of the specified length."""

        # Return a new tuple with the smallest value in the middle
        return [(sorted(lst[i:i + length])[1],
                 sorted(lst[i:i + length])[0],
                 sorted(lst[i:i + length])[2])
                    for i in range(0, len(lst), length)
               ]

    def _get_e_ads(self):
        e_ads = 0.0
        for ads_atoms in self._get_ads_atoms():
            ads_atoms.calc = self.calc_gcmc
            e_ads += ads_atoms.get_potential_energy()
        return e_ads

    def _setTqdm(self, start_step, totalsteps):
        self.pbar = tqdm.tqdm(total=totalsteps)
        self.pbar.n = start_step
        self.pbar.last_print_n = start_step  # Ensure the display updates correctly
        self.pbar.refresh()

    def _tqdmMD(self):
        self.pbar.update()

    def init_md(self, timestep, md_type="nptberendsen"):

        if not self.flex_ads:
            #  self._set_rigid_ads_atoms()
            self._set_rigid_triatomic_ads_atoms()

        self.atoms.calc = self.calc_md
        # Set initial velocities corresponding to T
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=self.T)
        trajectory = Trajectory(f'{self.results_dir}/trajectory_{self.P/bar}bar.traj', "a", self.atoms)

        if md_type.lower() == "nvt":
            # Define the Langevin dynamics
            self.dyn = Langevin(self.atoms,
                           timestep=timestep*fs,        # Timestep of 1 femtosecond
                           temperature_K=self.T,      # Target temperature in Kelvin
                           friction=0.01)          # Friction coefficient

        elif md_type.lower() == "npt":
            self.dyn = NPT(self.atoms,
                      timestep=timestep*fs,  # Timestep of 1 femtosecond
                      temperature_K=self.T,
                      externalstress=self.P,
                      ttime=25*fs,  # Thermostat coupling time
                      pfactor=0.6,)    # Barostat coupling factor

        elif md_type.lower() == "nptberendsen":
            self.dyn = NPTBerendsen(self.atoms,
                               timestep=timestep*fs,  # Timestep of 1 femtosecond
                               temperature_K=self.T,
                               pressure=self.P,
                               #  taut=0.5e3 *fs,
                               taut=1e2 *fs,
                               taup=5e2*fs,
                               compressibility=1e-7,
                              )
        elif md_type.lower() == "nptberendsen_inhomogeneous":
            self.dyn = Inhomogeneous_NPTBerendsen(self.atoms,
                               timestep=timestep*fs,  # Timestep of 1 femtosecond
                               temperature_K=self.T,
                               pressure=self.P,
                               #  taut=0.5e3 *fs,
                               taut=1e2 *fs,
                               taup=5e2*fs,
                               compressibility=1e-7,
                              )
        self.dyn.attach(trajectory.write, interval=self.interval)  # Write every step

    def _insertion_acceptance(self, e_trial):
        exp_value = self.beta * (self.e - e_trial)
        if exp_value > 100:
            return True
        elif exp_value < -100:
            return False
        else:
            acc = min(1, self.V*self.beta*self.fugacity/self.Z_ads * np.exp(exp_value))
            return np.random.rand() < acc

    def _deletion_acceptance(self, e_trial):
        exp_value = -self.beta * (e_trial - self.e)
        if exp_value > 100:
            return True
        else:
            acc = min(1, (self.Z_ads+1)/self.V/self.beta/self.fugacity * np.exp(exp_value))
            return np.random.rand() < acc

    def _insertion(self):
        self.n_trial_insertion += 1

        self.Z_ads += 1
        self.Z_ads_in_loop += 1

        atoms_trial = self.atoms.copy() + self.atoms_ads.copy()
        pos = atoms_trial.get_positions()
        pos[-self.n_ads:] = random_position(pos[-self.n_ads:], atoms_trial.get_cell())
        atoms_trial.set_positions(pos)
        if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, self.Z_ads-1):
            e_trial = 10**10 * kjmol
        else:
            atoms_trial.calc = self.calc_gcmc
            if self.flex_ads:
                e_trial = atoms_trial.get_potential_energy() * electronvolt - self._get_e_ads(atoms_trial) * electronvolt
            else:
                e_trial = atoms_trial.get_potential_energy() * electronvolt - self.e_ads * self.Z_ads
        if self._insertion_acceptance(e_trial):
            self.n_succ_insertion += 1
            self.n_tot_succ_steps += 1

            self.atoms.extend(atoms_trial[-self.n_ads:].copy())
            self.e = e_trial
        else:
            self.Z_ads -= 1
            self.Z_ads_in_loop -= 1

    def _deletion(self):
        self.n_trial_deletion += 1

        i_ads = np.random.randint(self.Z_ads)
        atoms_trial = self.atoms.copy()
        self.Z_ads -= 1
        self.Z_ads_in_loop -= 1
        del_atoms_idx = slice(self.n_frame + self.n_ads*i_ads, self.n_frame + self.n_ads*(i_ads+1))
        del atoms_trial[del_atoms_idx]
        #  del atoms_trial[self.n_frame + self.n_ads*i_ads : self.n_frame + self.n_ads*(i_ads+1)]
        atoms_trial.calc = self.calc_gcmc
        if self.flex_ads:
            e_trial = atoms_trial.get_potential_energy() * electronvolt - self._get_e_ads(atoms_trial) * electronvolt
        else:
            e_trial = atoms_trial.get_potential_energy() * electronvolt - self.e_ads * self.Z_ads
        if self._deletion_acceptance(e_trial):
            self.n_succ_deletion += 1
            self.e = e_trial

            del self.atoms[del_atoms_idx]
            self.n_tot_succ_steps += 1
        else:
            self.Z_ads += 1
            self.Z_ads_in_loop += 1

    def _translation(self):
        i_ads = np.random.randint(self.Z_ads)
        atoms_trial = self.atoms.copy()
        pos = atoms_trial.get_positions()
        pos[self.n_frame + self.n_ads*i_ads : self.n_frame + self.n_ads*(i_ads+1)] += 0.5 * (np.random.rand(3) - 0.5)
        atoms_trial.set_positions(pos)
        if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, i_ads):
            e_trial = 10**10 * kjmol
        else:
            atoms_trial.calc = self.calc_gcmc
            if self.flex_ads:
                e_trial = atoms_trial.get_potential_energy() * electronvolt - self._get_e_ads(atoms_trial) * electronvolt
            else:
                e_trial = atoms_trial.get_potential_energy() * electronvolt - self.e_ads * self.Z_ads
        acc = min(1, np.exp(-self.beta*(e_trial-self.e)))
        if acc > np.random.rand():
            self.atoms.set_positions(pos)
            self.e = e_trial
            self.n_tot_succ_steps += 1

    def _rotation(self):
        i_ads = np.random.randint(self.Z_ads)
        atoms_trial = self.atoms.copy()
        pos = atoms_trial.get_positions()
        pos[self.n_frame + self.n_ads*i_ads : self.n_frame + self.n_ads*(i_ads+1)] = _random_rotation(pos[self.n_frame + self.n_ads*i_ads : self.n_frame + self.n_ads*(i_ads+1)], circlefrac = 0.1)
        atoms_trial.set_positions(pos)
        if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, i_ads):
            e_trial = 10**10 * kjmol
        else:
            atoms_trial.calc = self.calc_gcmc
            if self.flex_ads:
                e_trial = atoms_trial.get_potential_energy() * electronvolt - self._get_e_ads(atoms_trial) * electronvolt
            else:
                e_trial = atoms_trial.get_potential_energy() * electronvolt - self.e_ads * self.Z_ads
        acc = min(1, np.exp(-self.beta*(e_trial-self.e)))
        if acc > np.random.rand():
            #  self.atoms = atoms_trial.copy()
            self.atoms.set_positions(pos)
            self.e = e_trial
            self.n_tot_succ_steps += 1

    def _set_gcmc(self, ngcmcsteps, nmcmoves):

        # setting for gcmc
        if not self.flex_ads:
            #  self.atoms.calc = self.calc_gcmc

            atoms_ads = self.atoms_ads.copy()
            atoms_ads_e = self.atoms_ads.copy()
            atoms_ads_e.calc = self.calc_gcmc
            self.e_ads = atoms_ads_e.get_potential_energy() * electronvolt # eV to Hartree
        #  e = atoms.get_potential_energy() * electronvolt # eV to Hartree

        self.uptake = []
        self.adsorption_energy = []

        self.fl_status = open(f"{self.results_dir}/status.csv", "w")
        print("Steps,trial_insertion, succ_insertion,trial_deletaion,succ_deletion", file=self.fl_status)
        self.fl_status.flush()

        self.n_trial_insertion = 0
        self.n_succ_insertion = 0
        self.n_trial_deletion = 0
        self.n_succ_deletion = 0
        self.n_tot_succ_steps = 0
        self.old_n_tot_succ_steps = 0
        self.ngcmcsteps = ngcmcsteps
        self.nmcmoves = nmcmoves
        self.ncycles = ngcmcsteps + nmcmoves

    def _run_gcmc(self):
        # to remove contraints after MD
        if not self.flex_ads:
            del self.atoms.constraints

        #get potential energy for GCMC after MD steps
        self.atoms.calc = self.calc_gcmc
        if self.flex_ads:
            self.e = self.atoms.get_potential_energy() * electronvolt - self._get_e_ads(self.atoms) * electronvolt
        else:
            self.e = self.atoms.get_potential_energy() * electronvolt - self.e_ads * self.Z_ads # eV to Hartree

        # set V after MD steps
        self.cell = np.array(self.atoms.get_cell())
        self.V = np.linalg.det(self.cell) * angstrom**3

        self.Z_ads_in_loop = 0
        #  print("Start GCMC for %d steps" %self.ncycles)
        #  for interation in tqdm.trange(self.ncycles):
        for interation in range(self.ncycles):
            # run GCMC exchanhes and MC moves
            ixm = int(random.uniform(0, self.ncycles)) + 1
            if ixm <= self.nmcmoves and self.Z_ads != 0:
                xmcmove = random.uniform(0, 1)
                if xmcmove < 0.5:
                    # Translation
                    self._translation()
                else:
                    # Rotation
                    self._rotation()
            else:
                xgcmc = random.uniform(0, 1)
                if xgcmc < 0.5:
                    # insertion
                    self._insertion()
                else:
                    # Deletion
                    if self.Z_ads != 0:
                        self._deletion()

            self.uptake.append(self.Z_ads)
            self.adsorption_energy.append(self.e)

            # save status at steps interval
            if self.n_tot_succ_steps % self.interval == 0 and self.old_n_tot_succ_steps != self.n_tot_succ_steps:
                write(f'{self.results_dir}/trajectory_{self.P/bar}bar.extxyz', self.atoms, append=True)
                print(f"{self.n_tot_succ_steps},{self.n_trial_insertion},{self.n_succ_insertion},{self.n_trial_deletion},{self.n_succ_deletion}", file=self.fl_status)

                self.fl_status.flush()
                self.old_n_tot_succ_steps = self.n_tot_succ_steps

            # save uptake and enrgirs at ever step in
            np.save(f'{self.results_dir}/uptake_{self.P/bar}bar.npy', np.array(self.uptake))
            np.save(f'{self.results_dir}/adsorption_energy_{self.P/bar}bar.npy', np.array(self.adsorption_energy))

        # assign velocites to new added atoms
        if self.Z_ads_in_loop > 0:
            velocities = self.atoms.get_velocities()
            add_ads_atoms = self.atoms[-self.n_ads*self.Z_ads_in_loop:].copy()
            MaxwellBoltzmannDistribution(add_ads_atoms, temperature_K=self.T)
            velocities[-self.n_ads*self.Z_ads_in_loop:] = add_ads_atoms.get_velocities()
            self.atoms.set_velocities(velocities)

        # set rigid constraints for MD if desired
        if not self.flex_ads:
            #  self._set_rigid_ads_atoms()
            self._set_rigid_triatomic_ads_atoms()

        #set model for MD
        self.atoms.calc = self.calc_md

    def run(self, totalsteps, nmdsteps, ngcmcsteps, nmcmoves):

        #  atoms = self.atoms_frame.copy()
        #  self.atoms = self.atoms_frame.copy()


        #  self._setTqdm(md_steps, totalsteps)
        self._setTqdm(0, totalsteps)
        self.dyn.attach(self._tqdmMD)

        self.dyn.run(nmdsteps)

        self._set_gcmc(ngcmcsteps, nmcmoves)
        self.dyn.attach(self._run_gcmc, interval=nmdsteps)

        self.dyn.run(totalsteps)


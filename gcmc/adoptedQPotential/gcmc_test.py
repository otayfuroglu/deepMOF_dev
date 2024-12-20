import numpy as np

from ase import Atoms
from ase.io import read, write
from molmod.units import *
from molmod.constants import *
from molmod.periodic import periodic
from utilities import _random_rotation, random_position, vdw_overlap
import tqdm

class AI_GCMC():
    def __init__(self, model, results_dir, interval, atoms_frame, atoms_ads, T, P, fugacity, device, vdw_radii):
        self.model = model
        self.results_dir = results_dir
        self.interval = interval
        self.atoms_frame = atoms_frame
        self.n_frame = len(self.atoms_frame)
        self.atoms_ads = atoms_ads
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

    def run(self, N):

        atoms = self.atoms_frame.copy()
        atoms.calc = self.model
        atoms_ads = self.atoms_ads.copy()
        atoms_ads_e = self.atoms_ads.copy()
        atoms_ads_e.calc = self.model
        e_ads = atoms_ads_e.get_potential_energy() * electronvolt # eV to Hartree
        e = atoms.get_potential_energy() * electronvolt # eV to Hartree

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

        for iteration in tqdm.trange(N):
            switch = np.random.rand()
            self.Z_ads += 1

            n = 0
            while n < 1:
                atoms_trial = atoms + self.atoms_ads.copy()
                pos = atoms_trial.get_positions()
                pos[-self.n_ads:] = random_position(pos[-self.n_ads:], atoms_trial.get_cell())
                atoms_trial.set_positions(pos)
                if vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, self.Z_ads-1):
                    e_trial = 10**10 * kjmol
                    print("INN")
                else:
                    n += 1
                    atoms_trial.calc = self.model
                    e_trial = atoms_trial.get_potential_energy() * electronvolt - e_ads * self.Z_ads
                    print(e - e_trial)
            quit()

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
                fl_status.flush()
                old_steps = steps

        np.save(f'{self.results_dir}/uptake_{self.P/bar}bar.npy', np.array(uptake))
        np.save(f'{self.results_dir}/adsorption_energy_{self.P/bar}bar.npy', np.array(adsorption_energy))



import sys, os
import numpy as np
from yaff import *
log.set_level(0)
from ase.data import vdw_radii
from ase import *
from ase.io import *
from glob import *
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import argparse

from ase import Atoms



class insertAds():
    def __init__(self, structure, ads, vdw_radii, pbc):
        self.structure = structure
        self.ads = ads
        self.rvecs = structure.cell.rvecs
        self.vdw = (vdw_radii - 0.35) * angstrom
        self.pbc = pbc
        #  self.n_ads = len(self.ads.numbers)
        #  self.fl_name = fl_name

    def _random_rotation(self, pos, circlefrac = 1.0):
        com = np.average(pos, axis=0)
        pos -= com
        randnums = np.random.uniform(size=(3,))
        theta, phi, z = randnums
        theta = theta * 2.0*circlefrac*np.pi  # Rotation about the pole (Z).
        phi = phi * 2.0*np.pi  # For direction of pole deflection.
        z = z * 2.0*circlefrac  # For magnitude of pole deflection.
        r = np.sqrt(z)
        Vx, Vy, Vz = V = (
            np.sin(phi) * r,
            np.cos(phi) * r,
            np.sqrt(2.0 - z)
            )
        st = np.sin(theta)
        ct = np.cos(theta)
        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
        M = (np.outer(V, V) - np.eye(3)).dot(R)
        pos = np.einsum('ib,ab->ia', pos, M)
        return pos + com

    def _random_translation(self, pos):
        rvecs = self.rvecs
        pos -= np.average(pos, axis=0)
        rnd = np.random.rand(3)
        new_cos = rnd[0]*rvecs[0] + rnd[1]*rvecs[1] + rnd[2]*rvecs[2]
        return pos + new_cos

    def random_position(self, pos):
        pos = self._random_rotation(pos)
        pos = self._random_translation(pos)
        return pos

    def vdw_overlap(self, system, pos_trial, num_trial):
        num, pos = system.numbers, system.pos
        for n_trial, p_trial in zip(num_trial, pos_trial):
            for n, p in zip(num, pos):
                vec = p_trial - p
                system.cell.mic(vec)
                d = np.linalg.norm(vec)
                if d < self.vdw[n_trial] + self.vdw[n]:
                    return True
        return False

    def load(self, n_trial, n_load):
        structure = self.structure
        num_ads = self.ads.numbers
        self.Z_ads = 0
        self.acc, self.tried = np.zeros(3), np.zeros(3)

        step = 0
        while step <= n_trial and n_load > 0:

            #  idx_rand = np.random.randint(1)
            # Insertion attempt
            pos_ads = self.ads.pos.copy()
            pos_ads = self.random_position(pos_ads)
            if not self.vdw_overlap(structure, pos_ads, num_ads):
                s_ads = System(numbers=num_ads, pos=pos_ads)
                s_ads.detect_bonds()
                structure = structure.merge(s_ads)

                self.Z_ads += 1
                self.acc[0] += 1
            self.tried[0] += 1
            if self.Z_ads == n_load: break
            step += 1

        numbers, pos = structure.numbers, structure.pos
        n_write = self.Z_ads
        atoms = Atoms(numbers=numbers, positions=pos/angstrom, cell=structure.cell.rvecs/angstrom, pbc=self.pbc)
        #  atoms.info["label"] = fl_name
        self.final_structure = atoms
        return n_write

    def write_output(self, out_path, append):
        write(out_path, self.final_structure, format='extxyz', append=append)


class insertMixAds(insertAds):

    def __init__(self, structure, ads, ads2, vdw_radii, pbc):
        self.ads2 = ads2
        super().__init__(structure, ads, vdw_radii, pbc)


    def _insert_trial(self, ads, num_ads):
        pos_ads = ads.pos.copy()
        pos_ads = self.random_position(pos_ads)
        if not self.vdw_overlap(self.structure, pos_ads, num_ads):
            s_ads = System(numbers=num_ads, pos=pos_ads)
            s_ads.detect_bonds()
            self.structure = self.structure.merge(s_ads)
            return True
        return False


    def load_mix(self, n_trial, n_load, ratio):
        #  structure = self.structure
        num_ads = self.ads.numbers
        self.Z_ads = 0
        num_ads2 = self.ads2.numbers
        self.Z_ads2 = 0
        #  self.acc, self.tried = np.zeros(3), np.zeros(3)

        step = 0
        while step <= n_trial and n_load > 0:

            #  idx_rand = np.random.randint(1)
            # Insertion attempt
            if np.random.rand() < ratio:
                if self._insert_trial(self.ads, num_ads):
                    self.Z_ads += 1
            else:
                if self._insert_trial(self.ads2, num_ads2):
                    self.Z_ads2 += 1
            if self.Z_ads + self.Z_ads2 == n_load: break
            step += 1

        numbers, pos = self.structure.numbers, self.structure.pos
        n_write = self.Z_ads + self.Z_ads2
        atoms = Atoms(numbers=numbers, positions=pos/angstrom, cell=self.structure.cell.rvecs/angstrom, pbc=self.pbc)
        self.final_structure = atoms
        return n_write

    def load_mix_fixed_nads(self, n_trial, nads, nads2):
        #  structure = self.structure
        num_ads = self.ads.numbers
        self.Z_ads = 0
        num_ads2 = self.ads2.numbers
        self.Z_ads2 = 0
        #  self.acc, self.tried = np.zeros(3), np.zeros(3)

        n_load = nads + nads2
        step = 0
        while step <= n_trial and n_load > 0:

            #  idx_rand = np.random.randint(1)
            # Insertion attempt
            if self.Z_ads < nads:
                if self._insert_trial(self.ads, num_ads):
                    self.Z_ads += 1
            if self.Z_ads2 < nads2:
                if self._insert_trial(self.ads2, num_ads2):
                    self.Z_ads2 += 1
            if self.Z_ads + self.Z_ads2 == n_load: break
            step += 1

        numbers, pos = self.structure.numbers, self.structure.pos
        n_write = self.Z_ads + self.Z_ads2
        atoms = Atoms(numbers=numbers, positions=pos/angstrom, cell=self.structure.cell.rvecs/angstrom, pbc=self.pbc)
        self.final_structure = atoms
        return n_write

    def print_ratio(self):
        print(f"ads1:ads2 = {self.Z_ads}:{self.Z_ads2}")


#
import numpy as np
from numpy import linalg

from ase.calculators.calculator import Calculator
from ase.utils import ff
from ase.utils.ff import Coulomb
from ase.neighborlist import NeighborList
from ase.io import read, write
from ase.units import kcal, kJ, mol

from openbabel import openbabel as ob


class ForceField(Calculator):
    implemented_properties = ['energy']
    nolabel = True

    def __init__(self, chargeCalc=None, rc=12, vdws=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        if chargeCalc is None:
            raise ImportError("Charge calculator must be defined!")

        self.chargeCalc = chargeCalc
        self.rc = rc

        if vdws is None:
            self.vdws = []
        else:
            self.vdws = vdws

    def _getCoulombs(self, atoms, nl):

        atoms = atoms.copy()
        charges = self.chargeCalc.get_charge(atoms).tolist()

        coulombs = []
        for i in range(len(atoms)):
            indices, offsets = nl.get_neighbors(i)
            for j in indices:
                if j > i:
                   coulombs.append(Coulomb(i, j, chargei=charges[i], chargej=charges[j]))
                #  print(Zi, Qi, Zj, Qj)
        return coulombs

    def _getVdW(self, atoms, nl):

        epsilons = {"MgC":0.0771, "OC":0.0567, "CC":0.0750, "HC":0.0485,
                    "MgO":0.1320, "OO":0.09705, "CO":0.1283, "HO":0.08311,
                   }
        sigmas = {"MgC":2.7457, "OC":2.959, "CC":3.115, "HC":2.685,
                  "MgO":2.870, "OO":3.084, "CO":3.240, "HO":2.8105,
                 }

        symbols = atoms.get_chemical_symbols()

        VdWs = []
        for i in range(len(atoms)):
            indices, offsets = nl.get_neighbors(i)
            for j in indices:
                atoms_pair = "".join([symbols[i],symbols[j]])
                if atoms_pair in list(epsilons.keys()):
                    #  print(atoms_pair, epsilons[atoms_pair], sigmas[atoms_pair])
                    if j > i:
                        vdw = ff.VdW(atomi=i, atomj=j, epsilonij=epsilons[atoms_pair], rminij=sigmas[atoms_pair])
                        VdWs.append(vdw)
        return VdWs

    def _atoms_to_obmol(self, atoms):
        """Convert an Atoms object to an OBMol object.
        Parameters
        ==========
        Input
            atoms: Atoms
        Return
            obmol: OBMol
        """

        obmol = ob.OBMol()
        for atom in atoms:
            a = obmol.NewAtom()
            a.SetAtomicNum(int(atom.number))
            a.SetVector(atom.position[0], atom.position[1], atom.position[2])
        # Automatically add bonds to molecule
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()
        return obmol

    def _get_VdWs_ob(self, atoms):
        obmol = self._atoms_to_obmol(atoms)
        ff = ob.OBForceField.FindForceField("uff")
        ff.Setup(obmol)
        return ff.E_VDW() * kJ / mol

    def coulomb_potential_energy_manuel(self, atoms):
        COULOMB_CONST = 14.399645478425116  # eV·Å/e² (approximate)


        atoms = atoms.copy()
        charges = self.chargeCalc.get_charge(atoms).tolist()
        num_atoms = len(atoms)

        nl = NeighborList([self.rc / 2] * num_atoms, self_interaction=False, bothways=True)
        nl.update(atoms)

        coulomb_energy = 0.0
        for i in range(num_atoms):
            neighbors, offsets = nl.get_neighbors(i)
            for j, offset in zip(neighbors, offsets):
                if j > i:  # Ensure no double counting of atom pairs
                    rij = rel_pos_pbc(atoms, i, j)
                    dij = linalg.norm(rij)
                    coulomb_energy += COULOMB_CONST * (charges[i] * charges[j]) / dij
        return coulomb_energy

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if system_changes:
            for name in ['energy', 'forces', 'hessian']:
                self.results.pop(name, None)

        nl = NeighborList([self.rc / 2] * len(atoms), self_interaction=False, bothways=True)
        nl.update(atoms)

        coulombs = self._getCoulombs(atoms, nl)
        vdws = self._getVdW(atoms, nl)

        if 'energy' not in self.results:
            energy = 0.0
            #  for vdw in vdws:
            #      i, j, e = ff.get_vdw_potential_value(atoms, vdw)
            #      energy += e * kcal / mol
            for coulomb in coulombs:
                i, j, e = ff.get_coulomb_potential_value(atoms, coulomb)
                energy += e

            #  energy += self.coulomb_potential_energy_manuel(atoms)
            energy += self._get_VdWs_ob(atoms)

            self.results['energy'] = energy


def rel_pos_pbc(atoms, i, j):
    """
    Return difference between two atomic positions,
    correcting for jumps across PBC
    """
    d = atoms.get_positions()[i, :] - atoms.get_positions()[j, :]
    if atoms.cell:
        g = linalg.inv(atoms.get_cell().T)
        f = np.floor(np.dot(g, d.T) + 0.5)
        d -= np.dot(atoms.get_cell().T, f).T
    return d

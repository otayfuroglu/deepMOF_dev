#
from numpy import linalg
import numpy as np

from ase.calculators.calculator import Calculator
#  from ase.utils import ff
#  from ase.utils.ff import Coulomb
from ase.neighborlist import NeighborList
from ase.io import read, write
from ase import units

from openbabel import openbabel as ob

#  from ase.utils import ff
#  from ase.utils.ff import Coulomb

# Coulomb constant in eV·Å/e² (using ASE's internal units)
COULOMB_CONST = 8.9875517873681764e9 * units.m * units.J / units.C / units.C

class Coulomb:
    def __init__(self, atomi, atomj, chargei=None, chargej=None, scale=1.0):
        self.atomi = atomi
        self.atomj = atomj
        if chargei is not None and chargej is not None:
            self.chargeij = scale * chargei * chargej * COULOMB_CONST
        else:
            raise NotImplementedError("not implemented combination"
                                      "of Coulomb parameters.")
        self.r = None



class ForceField(Calculator):
    implemented_properties = ['energy']
    nolabel = True

    def __init__(self, chargeCalc=None, rc=12, vdws=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        if chargeCalc is None:
            raise ImportError("Charge calculator must be defined!")

        self.chargeCalc = chargeCalc
        self.rc = rc

    def get_coulomb_potential_value(self, atoms, coulomb):
        i = coulomb.atomi
        j = coulomb.atomj

        rij = rel_pos_pbc(atoms, i, j)
        dij = linalg.norm(rij)
        v = coulomb.chargeij / dij
        coulomb.r = dij

        return i, j, v

    def _getCoulombs(self, atoms, nl):

        atoms = atoms.copy()
        #  charges = self.chargeCalc.get_charge(atoms).numpy()
        charges = atoms.arrays["HFPQ"] #NOTE
        #  charges = atoms.arrays["CHELPGPQ"] #NOTE
        #  charges = atoms.arrays["DDECPQ"] #NOTE

        coulombs = []
        for i in range(len(atoms)):
            indices, offsets = nl.get_neighbors(i)
            for j in indices:
                if j > i:
                   coulombs.append(Coulomb(i, j, chargei=charges[i], chargej=charges[j]))
                #  print(Zi, Qi, Zj, Qj)
        return charges, coulombs

    def coulomb_potential_energy_manuel(self, atoms, nl):

        atoms = atoms.copy()
        charges = self.chargeCalc.get_charge(atoms).numpy()
        #  charges = atoms.arrays["DDECPQ"] #NOTE

        coulomb_energy = 0.0
        num_atoms = len(atoms)
        # Loop over atoms and calculate Coulomb energy only with neighbors
        for i in range(num_atoms):
            neighbors, offsets = nl.get_neighbors(i)
            for j, offset in zip(neighbors, offsets):
                if j > i:  # Ensure no double counting of atom pairs
                    rij = rel_pos_pbc(atoms, i, j)
                    dij = linalg.norm(rij)
                    coulomb_energy += (COULOMB_CONST * charges[i] * charges[j]) / dij
        return coulomb_energy

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
        return obmol

    def _get_VdWs_ob(self, atoms):
        obmol = self._atoms_to_obmol(atoms)
        #  ff = ob.OBForceField.FindForceField("uff")
        #  ff = ob.OBForceField.FindForceField("gaff")
        #  ff = ob.OBForceField.FindForceField("mmff94")
        ff = ob.OBForceField.FindForceField("ghemical")
        ff.Setup(obmol)
        return ff.E_VDW() * units.kJ / units.mol

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if system_changes:
            for name in ['energy', 'forces', 'hessian']:
                self.results.pop(name, None)

        nl = NeighborList([self.rc] * len(atoms), self_interaction=False, bothways=True)
        nl.update(atoms)

        charges, coulombs = self._getCoulombs(atoms, nl)
        #  atoms.arrays["DDECPQ"] = charges

        if 'energy' not in self.results:
            energy = 0.0
            for coulomb in coulombs:
                i, j, e = self.get_coulomb_potential_value(atoms, coulomb)
            #  e = self.coulomb_potential_energy_manuel(atoms, nl)
                energy += e
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


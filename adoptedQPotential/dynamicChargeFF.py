import numpy as np

from ase.calculators.calculator import Calculator
from ase.utils import ff
from ase.utils.ff import Coulomb
from ase.neighborlist import NeighborList


class ForceField(Calculator):
    implemented_properties = ['energy', 'forces']
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

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if system_changes:
            for name in ['energy', 'forces', 'hessian']:
                self.results.pop(name, None)

        #NOTE dynamic charge

        charges = self.chargeCalc.get_charge(atoms).tolist()

        natoms = len(atoms)
        cell = atoms.cell
        nl = NeighborList([self.rc / 2] * natoms, self_interaction=False, bothways=True)
        nl.update(atoms)

        coulombs = []
        for i in range(natoms):
            indices, offsets = nl.get_neighbors(i)
            cells = np.dot(offsets, cell)
            for j, cell_j in zip(indices, cells):
               coulombs.append(Coulomb(i, j, chargei=charges[i], chargej=charges[j]))
                #  print(Zi, Qi, Zj, Qj)

        if 'energy' not in self.results:
            energy = 0.0
            for vdw in self.vdws:
                i, j, e = ff.get_vdw_potential_value(atoms, vdw)
                energy += e
            for coulomb in coulombs:
                i, j, e = ff.get_coulomb_potential_value(atoms, coulomb)
                if np.isnan(e):
                    #  print(charges[coulomb.atomi],charges[coulomb.atomj],coulomb.chargeij)
                    continue # NOTE
                energy += e
            self.results['energy'] = energy
        if 'forces' not in self.results:
            forces = np.zeros(3 * natoms)
            for vdw in self.vdws:
                i, j, g = ff.get_vdw_potential_gradient(atoms, vdw)
                limits = get_limits([i, j])
                for gb, ge, lb, le in limits:
                    forces[gb:ge] -= g[lb:le]
            for coulomb in coulombs:
                i, j, g = ff.get_coulomb_potential_gradient(atoms, coulomb)
                limits = get_limits([i, j])
                for gb, ge, lb, le in limits:
                    forces[gb:ge] -= g[lb:le]
            self.results['forces'] = np.reshape(forces, (natoms, 3))
        if 'hessian' not in self.results:
            hessian = np.zeros((3 * natoms, 3 * natoms))
            for vdw in self.vdws:
                i, j, h = ff.get_vdw_potential_hessian(atoms, vdw)
                limits = get_limits([i, j])
                for gb1, ge1, lb1, le1 in limits:
                    for gb2, ge2, lb2, le2 in limits:
                        hessian[gb1:ge1, gb2:ge2] += h[lb1:le1, lb2:le2]
            for coulomb in coulombs:
                i, j, h = ff.get_coulomb_potential_hessian(atoms, coulomb)
                limits = get_limits([i, j])
                for gb1, ge1, lb1, le1 in limits:
                    for gb2, ge2, lb2, le2 in limits:
                        hessian[gb1:ge1, gb2:ge2] += h[lb1:le1, lb2:le2]
            self.results['hessian'] = hessian

    def get_hessian(self, atoms=None):
        return self.get_property('hessian', atoms)


def get_limits(indices):
    gstarts = []
    gstops = []
    lstarts = []
    lstops = []
    for l, g in enumerate(indices):
        g3, l3 = 3 * g, 3 * l
        gstarts.append(g3)
        gstops.append(g3 + 3)
        lstarts.append(l3)
        lstops.append(l3 + 3)
    return zip(gstarts, gstops, lstarts, lstops)

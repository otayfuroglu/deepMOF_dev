#
import pynnp
from ase.calculators.calculator import Calculator, all_changes
from .ase_units import Units
import numpy as np
import os, shutil

from ase import units



class n2p2Calculator(Exception):
    pass


class n2p2Calculator(Calculator):
    """
    ASE calculator for n2p2 machine learning models.
    """
    energy = "energy"
    forces = "forces"
    implemented_properties = [energy, forces]

    def __init__(
        self,
        model_dir,
        best_epoch,
        energy=None,
        forces=None,
        energy_units="eV",
        length_units="Angstrom",
        #  forces_units="eV/Angstrom",
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)

        self.model_dir = model_dir
        self._prepareModel(best_epoch)


        self.model_energy = energy
        self.model_forces = forces

        # Convert to ASE internal units (energy=eV, length=A)
        self.energy_units = Units.unit2unit(energy_units, "eV")
        self.length_units = Units.unit2unit("Angstrom", length_units)
        self.forces_units = Units.unit2unit("/".join([energy_units, length_units]), "eV/Angstrom")

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)

        if self.calculation_required(atoms, properties):

            Calculator.calculate(self, atoms)

            with open('input.data', 'w') as input_data:
                input_data.write(self._atoms2n2p2(atoms))

            p = pynnp.Prediction()
            p.log.writeToStdout = False # n2p2 sdt ouptput is muted
            p.setup()
            p.readStructureFromFile()
            p.predict()
            s = p.structure

            self.model_energy = s.energy

            self.model_forces = np.zeros([len(atoms),3])
            for i,atom in enumerate(s.atoms):
                self.model_forces[i,:] = atom.f.r

            results = {}
            # Convert outputs to calculator format
            if self.model_energy is not None:
                results[self.energy] = (
                    self.model_energy * self.energy_units
                )  # ase calculator should return scalar energy

            if self.model_forces is not None:
                results[self.forces] = (
                   self.model_forces * self.forces_units
                )

            self.results = results

    def _atoms2n2p2(self, atoms):
        out_str = ''
        out_str += 'begin\n'
        out_str += 'comment created by ASE \n'

        n = len(atoms)

        if any(atoms.pbc):
            cell = atoms.get_cell() * self.length_units
            cell_template = 'lattice {:10.6f} {:10.6f} {:10.6f}\n'
            for c in cell:
                out_str += cell_template.format(c[0], c[1], c[2])

        atom_template = 'atom {:10.6f} {:10.6f} {:10.6f} {:2s} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}\n'
        forces = np.zeros([n,3])
        for a in atoms:
            #  force = forces[a.index]
            position = a.position * self.length_units
            out_str += atom_template.format(position[0], position[1], position[2],
                                            #  a.symbol, 0.0, 0.0, force[0], force[2], force[3])
                                            a.symbol, 0.0, 0.0, 0.0, 0.0, 0.0)

        energy = 0
        out_str += 'energy {:10.6f}\n'.format(energy)
        out_str += 'charge 0.0\n'
        out_str += 'end\n'

        return out_str

    def _prepareModel(self, best_epoch):
        os.system(f"cp {self.model_dir}/input.nn ./")
        os.system(f"cp {self.model_dir}/scaling.data ./")
        weights_files = [item for item in os.listdir(self.model_dir) if "weights" in item]
        best_weights_files = [item for item in weights_files if int(item.split(".")[-2]) == best_epoch]
        assert len(best_weights_files) != 0, "Erro: NOT FOUND best epoch number"
        for best_weights_file in best_weights_files:
            os.system(f"cp {self.model_dir}/{best_weights_file} ./{best_weights_file[:11]}.data")



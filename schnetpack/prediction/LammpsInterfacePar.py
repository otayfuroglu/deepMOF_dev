from lammps_interface.lammps_main import LammpsSimulation
from lammps_interface.structure_data import from_CIF, write_CIF
import sys


class Parameters:
    def __init__(self):
        # File options
        self.output_cif = False
        self.output_raspa = False

        # Force field options
        self.force_field = 'UFF'
        self.mol_ff = None
        self.h_bonding = False
        self.dreid_bond_type = 'harmonic'
        self.fix_metal = False

        # Simulation options
        self.minimize = True
        self.bulk_moduli = False
        self.thermal_scaling = False
        self.thermal_anneal = False
        self.relax = False
        self.npt = False
        self.nvt = False
        self.cutoff = 12.5
        self.replication = None
        self.orthogonalize = False
        self.random_vel = False
        self.dump_dcd = 0
        self.dump_xyz = 0
        self.dump_lammpstrj = 0
        self.restart = False

        # Parameter options
        self.tol = 0.4
        self.neighbour_size = 5
        self.iter_count = 10
        self.max_dev = 0.01
        self.temp = 298.0
        self.pressure = 1.0
        self.nprodstp = 200000
        self.neqstp = 200000

        # Molecule insertion options
        #  self.insert_molecule = ""
        #  self.deposit = 0

    def show(self):
        for v in vars(self):
            print('%-15s: %s' % (v, getattr(self, v)))


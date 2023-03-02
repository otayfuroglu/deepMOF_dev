
from runnerAseInterFace import runnerCalculator

from ase.db import connect
from ase import Atoms

import os, shutil

os.chdir("/kernph/tayfur0000/works/alanates/runner/test")

calculator = runnerCalculator(model_dir="/kernph/tayfur0000/works/alanates/runner/forces_scalingDefault_alldata_single_task",
                            best_epoch=40,
                            energy_units="Hartree",
                            length_units="Bohr",
                           )

db = connect("/kernph/tayfur0000/works/alanates/n2p2/zeta_12416_frate10_alldata_24_task/test.db")
row = db.get(1)

#  atoms = row.toatoms()
atoms = Atoms(symbols=row.symbols, positions=row.positions, cell=row.cell)
atoms.pbc = True
atoms.set_calculator(calculator)

runner_energy = atoms.get_potential_energy()
runner_forces = atoms.get_forces()

print(runner_energy)

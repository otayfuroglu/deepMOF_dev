
from calculationWithASE import N2P2Calculator
from ase.build import molecule
atoms = molecule('H2O')

cacultor = N2P2Calculator(nnp_dir="./H2O_RPBE-D3")

atoms.set_calculator(cacultor)
print(atoms.get_potential_energy())
print(atoms.get_forces())




from dynamicChargeFF import ForceField
from ChargePredict import ChargeCalculator
from ase.io import read



atoms = read("./s1.extxyz")

nn1 = 200
nn2 = 160
chargeCalc = ChargeCalculator(hidden_size1=nn1,hidden_size2=nn2)

atoms.calc = ForceField(chargeCalc)
print(atoms.get_potential_energy())

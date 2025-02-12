from ase.io import read
from openbabel import openbabel as ob

from ase.units import kJ, mol



def _atoms_to_obmol(atoms):
    """Convert an Atoms object to an OBMol object.

    Parameters
    ==========
    Input
        atoms: Atoms
    Return
        obmol: OBMol

    """

    charges = atoms.arrays["DDECPQ"]
    obmol = ob.OBMol()
    for atom, charge in zip(atoms, charges):
        a = obmol.NewAtom()
        a.SetAtomicNum(int(atom.number))
        a.SetVector(atom.position[0], atom.position[1], atom.position[2])
        #  a.SetPartialCharge(charge)

    # Automatically add bonds to molecule
    obmol.ConnectTheDots()
    obmol.PerceiveBondOrders()

    for charge, atom in zip(charges, ob.OBMolAtomIter(obmol)):
        atom.SetPartialCharge(charge)
        atom.GetPartialCharge()



    #  for atom in ob.OBMolAtomIter(obmol):
        #  print(atom.GetPartialCharge())

    return obmol


def _get_energy_ob(obmol):

    #  ff = ob.OBForceField.FindForceField("mmff94")
    ff = ob.OBForceField.FindForceField("uff")
    #  success = ff.Setup(obmol)
    if ff.Setup(obmol):

        #  e = ff.Energy()
        #  print(ff.GetUnit())
        print(ff.E_Electrostatic() * kJ / mol)
        print(ff.E_VDW() * kJ / mol)
        #  quit()
        #  ff.Energy()  * kJ / mol # in eV

atoms = read("./test_charged.extxyz")
obmol = _atoms_to_obmol(atoms)
for atom in ob.OBMolAtomIter(obmol):
    print(atom.GetPartialCharge())

_get_energy_ob(obmol)
#  print(_get_energy_ob(atoms)))


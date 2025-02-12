from openbabel import openbabel

# Create an OBConversion object to read molecular formats
obConversion = openbabel.OBConversion()
obConversion.SetInAndOutFormats("xyz", "xyz")

# Create an OBMol object for the molecule
mol = openbabel.OBMol()

# Read in the CO2 molecule (or use any other molecule in XYZ format)
xyz_data = """3
CO2
C       0.00000    0.00000    0.00000
O       1.16000    0.00000    0.00000
O      -1.16000    0.00000    0.00000
"""
obConversion.ReadString(mol, xyz_data)

# Assign Gasteiger charges (UFF works with Gasteiger charges)
mol.AddHydrogens()  # UFF expects hydrogens; make sure the molecule is properly protonated
openbabel.OBChargeModel.FindType("gasteiger").ComputeCharges(mol)

# Initialize the UFF force field
uff = openbabel.OBForceField.FindForceField("UFF")

# Check if UFF is available and can be set up
if uff.Setup(mol):
    # Optimize the molecule (optional)
    uff.ConjugateGradients(500, 1.0e-4)

    # Calculate the energies
    total_energy = uff.Energy()  # Total energy
    electrostatic_energy = uff.E_Electrostatic()  # Electrostatic energy

    # Output the results
    print(f"Total Energy (UFF): {total_energy:.4f} kJ/mol")
    print(f"Electrostatic Energy (UFF): {electrostatic_energy:.4f} kJ/mol")

else:
    print("Failed to set up UFF for the molecule.")


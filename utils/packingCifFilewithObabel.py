import openbabel

def cif_to_xyz_babel(xyz_path, cif_path):

    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("cif", "xyz")
    myOBMol = openbabel.OBMol()
    obConversion.ReadFile(myOBMol, cif_path)
    fillUC = openbabel.OBOp.FindType("fillUC")
    fillUC.Do(myOBMol, "strict")
    obConversion.WriteFile(myOBMol, xyz_path)


def cif_to_cif_babel(new_cif_path, cif_path):

    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("cif", "cif")
    myOBMol = openbabel.OBMol()
    obConversion.ReadFile(myOBMol, cif_path)
    fillUC = openbabel.OBOp.FindType("fillUC")
    fillUC.Do(myOBMol, "strict")
    obConversion.WriteFile(myOBMol, new_cif_path)


xyz_path = "deneme.xyz"
cif_path = "./IRMOF-10.cif"
new_cif_path = "./obabel_IRMOF-10.cif"

#  cif_to_xyz_babel(xyz_path, cif_path)
cif_to_cif_babel(new_cif_path, cif_path)

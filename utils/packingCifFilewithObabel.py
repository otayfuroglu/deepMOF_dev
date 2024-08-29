from openbabel import openbabel
import os
import argparse


def cif_to_cif_babel(cif_path, out_format):

    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("cif", out_format)
    myOBMol = openbabel.OBMol()
    obConversion.ReadFile(myOBMol, cif_path)
    fillUC = openbabel.OBOp.FindType("fillUC")
    fillUC.Do(myOBMol, "strict")

    split_path = cif_path.split("/")
    file_name = split_path[-1].replace("cif", out_format)
    new_file_name = "filled_" + file_name

    if len(split_path) == 1:
        new_cif_path = new_file_name
    else:
        new_cif_path = "/".join(split_path[:-1]) + "/" + new_file_name
    obConversion.WriteFile(myOBMol, new_cif_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="help")
    parser.add_argument(
        "-cifPath", "--cifPath", type=str,
        required=True, help="give hdf5 file base")
    parser.add_argument(
        "-outFormat", "--outFormat", type=str,
        required=True, help="give hdf5 file base")

    args = parser.parse_args()
    cif_to_cif_babel(args.cifPath, args.outFormat)

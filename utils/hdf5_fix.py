#! /truba/home/yzorlu/miniconda3/bin/python
#
import h5py
import argparse
import ntpath

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-h5py_file", "--file_path",
                    type=str, required=True,
                    help="give full molecule path ")
args = parser.parse_args()

file_path = args.file_path
f = h5py.File(file_path, "r", libver="latest", swmr=True)

directory, name = ntpath.split(file_path)
fixed_file_path = "%s/fixed_%s" %(directory, name)

if __name__ == "__main__":
    fixed_f = h5py.File(fixed_file_path, "w", libver="latest")
    for key in f.keys():
        print(key)
        f.copy(key, fixed_f)
    fixed_f.close()





















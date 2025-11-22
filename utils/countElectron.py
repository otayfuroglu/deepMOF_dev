#
from ase.io import read
import argparse

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-file_path", type=str, required=True)

args = parser.parse_args()

atoms = read(args.file_path)
print(sum(atoms.get_atomic_numbers()))



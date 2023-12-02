from ase.io import read
import tqdm
import argparse


def aseDB2n2p2(atoms_list):
    fl = open("input.data", "w")
    for atoms in tqdm.tqdm(atoms_list):
        atoms_prep_list = [["begin"],]
        atom_template = 'atom {:10.6f} {:10.6f} {:10.6f} {:2s} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}'
        atoms_prep_list += [[atom_template.format(
            position[0], position[1], position[2],
            symbol, 0.0, 0.0,
            forces[0], forces[1], forces[2])]
            for position, symbol, forces in zip(
                atoms.get_positions().tolist(), atoms.get_chemical_symbols(), atoms.get_forces().tolist())]
        atoms_prep_list += [["energy ", atoms.get_potential_energy()], ["charge 0.0"], ["end"]]
        for line in atoms_prep_list:
            for item in line:
                fl.write(str(item))
            fl.write("\n")
    fl.close()

def randAseDB2n2p2(atoms_list, N):
    import random
    rand_list = random.sample(range(len(atoms_list)), N)
    fl = open(f"input.data.rand{N}", "w")
    for i, atoms in enumerate(tqdm.tqdm(atoms_list)):
        if i in rand_list:
            # remove random value for efficiency
            rand_list.remove(i)
            atoms_prep_list = [["begin"],]
            atom_template = 'atom {:10.6f} {:10.6f} {:10.6f} {:2s} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}'
            atoms_prep_list += [[atom_template.format(
                position[0], position[1], position[2],
                symbol, 0.0, 0.0,
                forces[0], forces[1], forces[2])]
                for position, symbol, forces in zip(
                    atoms.get_positions().tolist(), atoms.get_chemical_symbols(), atoms.get_forces().tolist())]
            atoms_prep_list += [["energy ", atoms.get_potential_energy()], ["charge 0.0"], ["end"]]
            for line in atoms_prep_list:
                for item in line:
                    fl.write(str(item))
                fl.write("\n")
    fl.close()


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-extxyz_path", type=str, required=True)
parser.add_argument("-N", type=int, default=0, required=False)
args = parser.parse_args()


atoms_list = read(args.extxyz_path, index=":")
N = args.N

if N == 0:
    aseDB2n2p2(atoms_list)
else:
    randAseDB2n2p2(atoms_list, N)

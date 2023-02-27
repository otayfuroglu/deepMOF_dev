from ase.db import connect
import tqdm
import argparse


def aseDB2n2p2(db):
    db = db.select()
    fl = open("input.data", "w")
    for row in tqdm.tqdm(db):
        #  atoms_prep_list = [["begin"], ["comment ", row.name]]
        atoms_prep_list = [["begin"],]
        atom_template = 'atom {:10.6f} {:10.6f} {:10.6f} {:2s} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}'
        atoms_prep_list += [[atom_template.format(
            position[0], position[1], position[2],
            symbol, 0.0, 0.0,
            forces[0], forces[1], forces[2])]
            for position, symbol, forces in zip(
                row.positions.tolist(), row.symbols, row.forces.tolist())]
        atoms_prep_list += [["energy ", row.energy], ["charge 0.0"], ["end"]]
        for line in atoms_prep_list:
            for item in line:
                fl.write(str(item))
            fl.write("\n")
    fl.close()

def randAseDB2n2p2(db, N):
    import random
    rand_list = random.sample(range(db.count()), N)
    db = db.select()
    fl = open(f"input.data.rand{N}", "w")
    for i, row in enumerate(tqdm.tqdm(db)):
        if i in rand_list:
            # remove random value for efficiency
            rand_list.remove(i)
            atoms_prep_list = [["begin"], ["comment ", row.name]]
            atom_template = 'atom {:10.6f} {:10.6f} {:10.6f} {:2s} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}'
            atoms_prep_list += [[atom_template.format(
                position[0], position[1], position[2],
                symbol, 0.0, 0.0,
                forces[0], forces[1], forces[2])]
                for position, symbol, forces in zip(
                    row.positions.tolist(), row.symbols, row.forces.tolist())]
            atoms_prep_list += [["energy ", row.energy], ["charge 0.0"], ["end"]]
            for line in atoms_prep_list:
                for item in line:
                    fl.write(str(item))
                fl.write("\n")
    fl.close()


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-db", type=str, required=True)
parser.add_argument("-N", type=int, default=0, required=False)
args = parser.parse_args()


db = connect(args.db)
N = args.N

if N == 0:
    aseDB2n2p2(db)
else:
    randAseDB2n2p2(db, N)

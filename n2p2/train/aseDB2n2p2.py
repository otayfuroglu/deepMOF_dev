from ase.db import connect
import tqdm


def aseDB2n2p2(db):
    db = db.select()
    fl = open("input.data", "w")
    for row in tqdm.tqdm(db):
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



db_path = "../../../deepMOF/HDNNP/prepare_data/workingOnDataBase/"\
    + "nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling_IRMOFseries1_4_6_7_10_merged_50000_ev.db"
    #  + "nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling.db" # for  just MOF5
db = connect(db_path)
#  aseDB2n2p2(db)
randAseDB2n2p2(db, 30000)

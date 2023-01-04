from ase.db import connect
from ase import units
import tqdm
import random


u = units.create_units("2014")
EV2HARTREE = 1.0 / u["Hartree"]
ANG2BOHR = 1.0 / u["Bohr"]

def getWriteRowInfo(row, fl):
    atoms_prep_list = [["begin"], ["comment ", row.name]]
    atom_template = 'atom {:10.6f} {:10.6f} {:10.6f} {:2s} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}'
    atoms_prep_list += [[atom_template.format(
        position[0], position[1], position[2],
        symbol, 0.0, 0.0,
        forces[0], forces[1], forces[2])]
        for position, symbol, forces in zip(
            (row.positions * ANG2BOHR).tolist(), row.symbols, (row.forces * (EV2HARTREE/ANG2BOHR)).tolist())]
            #  row.positions.tolist(), row.symbols, row.forces.tolist())]
    atoms_prep_list += [["energy ", row.energy * EV2HARTREE], ["charge 0.0"], ["end"]]
    for line in atoms_prep_list:
        for item in line:
            fl.write(str(item))
        fl.write("\n")

def aseDBrunner_v2(db, fl_name="input.data", N=0):
    rand_list = random.sample(range(db.count()), N)
    db = db.select()
    fl = open(fl_name, "w")
    for i, row in enumerate(tqdm.tqdm(db)):
        if len(rand_list) != 0:
            if i in rand_list:
                getWriteRowInfo(row, fl)
        else:
            getWriteRowInfo(row, fl)
    fl.close()


db_path = "../../geom_files/"\
    + "nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling_IRMOFseries1_4_6_7_10_merged_50000_ev.db"
    #  + "nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling.db" # for  just MOF5
db = connect(db_path)
aseDBrunner_v2(db, N=1000)
#  randAseDB2n2p2(db, 30000)



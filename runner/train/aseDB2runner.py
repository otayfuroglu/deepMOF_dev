from ase.db import connect
from ase import units
import pandas as pd
import tqdm
import random
from multiprocessing import Pool


u = units.create_units("2014")
EV2HARTREE = 1.0 / u["Hartree"]
ANG2BOHR = 1.0 / u["Bohr"]


def getWriteRowInfo(idx):
    row = db.get(idx)
    if conf_type:
        if conf_type not in row.name:
            return 0
    atoms_prep_list = [["begin"], ["comment ", row.name]]
    atom_template = 'atom {:10.6f} {:10.6f} {:10.6f} {:2s} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}'
    atoms_prep_list += [[atom_template.format(
        position[0], position[1], position[2],
        symbol, 0.0, 0.0,
        forces[0], forces[1], forces[2])]
        for position, symbol, forces in zip(
            (row.positions * ANG2BOHR).tolist(), row.symbols,
            (row.forces * (EV2HARTREE/ANG2BOHR)).tolist())]
            #  row.positions.tolist(), row.symbols, row.forces.tolist())]
    atoms_prep_list += [["energy ",
                         row.energy * EV2HARTREE], ["charge 0.0"], ["end"]]

    return atoms_prep_list, row.energy * EV2HARTREE / len(row["numbers"])


def run_func():
    for result in tqdm.tqdm(
        pool.imap_unordered(
            func=getWriteRowInfo, iterable=iterable), total=n_db):
        if result != 0:
            for line in result[0]:
                for item in line:
                    fl.write(str(item))
                fl.write("\n")
                fl.flush()
            energies.append(result[1])


#  db_path = "../../geom_files/"\
#      + "nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling_IRMOFseries1_4_6_7_10_merged_50000_ev.db"
#      #  + "nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling.db" # for  just MOF5
db_path = "../../../deepMOF/HDNNP/prepare_data/workingOnDataBase/"\
    + "nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling_IRMOFseries1_4_6_7_10_merged_50000_ev.db"
db = connect(db_path)
n_db = len(db)

conf_type = "mof5_f1"
#  conf_type = "irmofseries7_f1"
N = 1000

# if you set N >0 which is number of data point as intger
# will execute rondum selection
fl_name = "input.data"
names = []
energies = []
with open(fl_name, "w") as fl:
    with Pool(processes=28) as pool:
        if N != 0:
            while len(energies) <= N:
                iterable = random.sample(range(1, n_db+1), N)
                run_func()
        else:
            iterable = range(1, n_db+1)
            run_func()

df = pd.DataFrame()
df["Energy(Hartree/atom)"] = energies
df.to_csv("energies.csv")

from ase.db import connect

db_path = "../../../deepMOF/HDNNP/prepare_data/workingOnDataBase/"\
    + "nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling_IRMOFseries1_4_6_7_10_merged_50000_ev.db"
db = connect(db_path).select()

fl = open("input.data", "w")
for row in db:
    atoms_prep_list = [["begin"], ["comment\t", row.name]]
    atoms_prep_list += [["atom"] + coord + [sym] + [0.0, 0.0] + forces
                        for coord, sym, forces in zip(row.positions.tolist(),
                                                      row.symbols,
                                                      row.forces.tolist())]
    atoms_prep_list += [["energy", row.energy], ["charge 0.0"], ["end"]]
    for line in atoms_prep_list:
        for item in line:
            fl.write(str(item))
            fl.write("\t")
        fl.write("\n")
fl.close()


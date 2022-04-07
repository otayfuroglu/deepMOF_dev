#
from multiprocessing import Pool
from ase.db import connect
from ase import units
# from ase.io import read
from schnetpack import AtomsData
import os
import numpy as np
import pandas as pd
import tqdm
import argparse
import torch



def get_fmax_idx(forces):
    """
    Args:
    forces(3D torch tensor)

    retrun:
    maximum force conponent indices (zero 1D torch tensor)
    """

    if not torch.is_tensor(forces):
        forces = torch.from_numpy(forces) # if numpy array convert to 3d tensor
    else:
        forces = forces.squeeze(0)

    abs_forces = forces.abs()
    abs_idxs = (abs_forces==torch.max(abs_forces)).nonzero().squeeze(0) # get index of max value in abs_forces.
    if len(abs_idxs.shape) > 1: # if there are more than one max value
        abs_idxs = abs_idxs[0]  # select just one
    return abs_idxs


def get_fmax_componentFrom_idx(forces, fmax_component_idx):
    """
    xxx
    """

    if not torch.is_tensor(forces):
        forces = torch.from_numpy(forces) # if numpy array convert to 3d tensor
    else:
        forces = forces.squeeze(0)

    return forces[fmax_component_idx[0], fmax_component_idx[1]].item()


class GetPrintDB:

    def __init__(self, db_path, BASE_DIR):
        self.BASE_DIR = BASE_DIR
        self.UNIT = None
        self.db_path = db_path
        self.db = None
        self.notInListFileBase = None
        self.notInListFragBase = None
        self.notInFragNum = None

    def _loadAseDB(self):
        self.db = connect(self.db_path).select()

    def _loadSchDB(self):
        self.db = AtomsData(self.db_path)

    def _getSchDBWithPath(self, db_path, properties):
        return AtomsData(db_path, load_only=properties)

    def _setUnit(self, UNIT):
        self.UNIT = UNIT

    def calculatedFiles2csv(self, out_csv_path, frag_base=None):
        """
        Args:
            out_csv_path: full path of csv file where is saved file_names
            file_base(optional): if we wish extract specific fragment, we use this option
        """
        self._loadAseDB()
        df = pd.DataFrame()
        if frag_base:
            name_list = [row["name"] for row in self.db if frag_base in row["name"]]
        else:
            name_list = [row["name"] for row in self.db]
        df["FileNames"] = name_list
        df.to_csv(out_csv_path, index=None)

    def energiesFmax2csvASE(self, out_csv_path):
        """
        return values in eV !!!
        """
        self._loadAseDB()
        df = pd.DataFrame()
        data_list = np.array([[row["name"], row["energy"],
                               row.fmax] for row in self.db])
        df["FileNames"] = data_list[:, 0]
        df["Energy"] = data_list[:, 1]
        df["Fmax"] = data_list[:, 2]
        df.to_csv(out_csv_path, index=None)

    def _energiesFmax2csv(self, idx):
        """
        xxx
        """
        file_base = self.db.get_name(idx)
        energy = self.db[idx]["energy"].item()

        forces = self.db[idx]["forces"]
        # first get fmax indices than get qm_fmax and schnet fmax component
        fmax_component_idx = get_fmax_idx(forces)
        fmax_component = get_fmax_componentFrom_idx(forces, fmax_component_idx)
        fmax_comps = fmax_component

        return file_base, energy, fmax_comps

    def energiesFmax2csv(self, num_processes, out_csv_path):

        self._loadSchDB()
        lenDB = len(self.db)

        df = pd.DataFrame()

        file_names = []
        energies = []
        fmax_comps = []

        # implementation of  multiprocessor in tqdm.
        # Ref.https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
        pool = Pool(processes=num_processes)
        for result in tqdm.tqdm(
            pool.imap_unordered(func=self._energiesFmax2csv, iterable=range(lenDB)), total=lenDB):
            if result:
                file_names.append(result[0])
                energies.append(result[1])
                fmax_comps.append(result[2])

        df["FileNames"] = file_names
        df["Energy"] = energies
        df["Fmax"] = fmax_comps
        df.to_csv(out_csv_path, index=None)

    def _getPartOfDB(self, properties, partOf_keyword, antiKeyword, N=None):
        self._loadSchDB()

        if N is None:
            N = len(self.db)

        property_list = []
        atoms_list = []
        name_list = []
        counter = 0

        for i in tqdm.trange(len(self.db)):
            file_base = self.db.get_name(i)
            if partOf_keyword not in file_base or antiKeyword in file_base:
                print(file_base)
                continue

            property_values = []
            for propert in properties:
                mol = self.db.get_atoms(i)
                target_propert = self.db[i][propert]
                target_propert = np.array(target_propert, dtype=np.float)
                property_values.append(target_propert)
            # combine two lists into a dictionary
            property_dict = dict(zip(properties, property_values))
            atoms_list.append(mol)
            name_list.append(file_base)
            property_list.append(property_dict)
            counter += 1
            if counter % 100 == 0:
                print("Counter: ", counter)
            if counter == N:
                return atoms_list, name_list, property_list
        return atoms_list, name_list, property_list

    def partOfDB2NewDB(self, new_db_path):
        if os.path.exists(new_db_path):
            os.remove(new_db_path)
        #  atoms, properties = db.get_properties(0)
        #  properties = [propert for propert in
        #                properties.keys() if "_" not in propert]
        #  properties = ["energy", "forces", "dipole_moment"]
        new_db = AtomsData(new_db_path,
                           available_properties=properties)
        antiKeyword = "mof5_f4_outOfSFGeom_600K_24923"
        for partOf_keyword in ["mof5_f"]:  # "mof5_new_f2", "mof5_new_f3"]:
            atoms_list, name_list, property_list = self._getPartOfDB(
                properties, partOf_keyword,
                antiKeyword)
            new_db.add_systems(atoms_list, name_list, property_list)

    def _selectDB(self, i):
        self._loadSchDB()
        #  atoms, properties = db.get_properties(0)
        #  properties = [propert for propert in
        #                properties.keys() if "_" not in propert]
        #  properties = ["energy", "forces", "dipole_moment"]

        # to from csv file
        self.notInListFileBase = pd.read_csv("./removeFileNamesFronDB.csv")["FileNames"].to_list()
        #  self.notInListFragBase = ["irmofseries5"]
        #  self.notInFragNum = ["f1", "f2", "f3", "f4", "f5"]
        #  self.notInListFragBase = ["irmofseries1", "irmofseries6", "irmofseries7",
        #      + "irmofseies8", "irmofseries12", "irmofseries14", "irmofseries16"]

        # NOTE: set false if you don't know what you do
        remove = False

        file_base = self.db.get_name(i)
        frag_base = file_base.split("_")[0]
        frag_num = file_base.split("_")[1]

        if self.notInListFragBase:
            if frag_base in self.notInListFragBase:
                print(file_base)
                if remove:
                    # remove this coord file
                    try:
                        os.remove("%s/outOfSFGeomsIRMOFs%s/%s.xyz" %(self.BASE_DIR, mof_num, file_base))
                        return None
                    except:
                        print("%s.xyz file not found in outOfSFGeomsIRMOFs%s" % (file_base, mof_num))
                        return None
                return None

        elif self.notInListFileBase:
            if file_base in self.notInListFileBase:
                print(file_base)
                if remove:
                    # remove this coord file
                    try:
                        os.remove("%s/outOfSFGeomsIRMOFs%s/%s.xyz" %(self.BASE_DIR, mof_num, file_base))
                        return None
                    except:
                        print("%s.xyz file not found in outOfSFGeomsIRMOFs%s" % (file_base, mof_num))
                        return None
                return None

        elif self.notInFragNum:
            if frag_num in self.notInFragNum:
                print(file_base)
                if remove:
                    # remove this coord file
                    try:
                        os.remove("%s/outOfSFGeomsIRMOFs%s/%s.xyz" %(self.BASE_DIR, mof_num, file_base))
                        return None
                    except:
                        print("%s.xyz file not found in outOfSFGeomsIRMOFs%s" % (file_base, mof_num))
                        return None
                return None

        property_values = []
        for propert in properties:
            mol = self.db.get_atoms(i)
            target_propert = self.db[i][propert]
            target_propert = np.array(target_propert, dtype=np.float32)
            property_values.append(target_propert)
        # combine two lists into a dictionary
        property_dict = dict(zip(properties, property_values))
        return mol, file_base, property_dict

    def selectedDB2DB(self, num_processes, new_db_path):

        self._loadSchDB()
        if os.path.exists(new_db_path):
            os.remove(new_db_path)
        #  atoms, properties = db.get_properties(0)
        #  properties = [propert for propert in
        #                properties.keys() if "_" not in propert]
        #  properties = ["energy", "forces", "dipole_moment"]
        new_db = AtomsData(new_db_path,
                           available_properties=properties)
        lenDB = len(self.db)
        property_list = []
        atoms_list = []
        name_list = []

        # for random selection from given databases
        #  idxs = np.random.randint(0, lenDB, 25000).tolist()
        idxs = range(lenDB)
        # implementation of  multiprocessor in tqdm.
        # Ref.https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
        pool = Pool(processes=num_processes)
        for result in tqdm.tqdm(
            pool.imap_unordered(func=self._selectDB, iterable=idxs),
            total=len(idxs)):
            if result:
                atoms_list.append(result[0])
                name_list.append(result[1])
                property_list.append(result[2])
        new_db.add_systems(atoms_list, name_list, property_list)

    def _mergeDataBases(self, idx):
        #  properties = ["energy", "forces"]  # , "dipole_moment"]
        second_db = self._getSchDBWithPath(second_db_path, properties)
        file_base = second_db.get_name(idx)
        property_values = []
        for propert in properties:
            mol = second_db.get_atoms(idx)
            target_propert = second_db[idx][propert]
            target_propert = np.array(target_propert, dtype=np.float32)
            property_values.append(target_propert)

            # combine two lists into a dictionary
            property_dict = dict(zip(properties, property_values))
        return mol, file_base, property_dict

    def mergeDataBases(self, num_processes, second_db_path, merged_db_path):
        import shutil

        if os.path.exists(merged_db_path):
            os.remove(merged_db_path)

        #  properties = ["energy", "forces"]  # , "dipole_moment"]
        # copy db which has above properties
        shutil.copy2(self.db_path, merged_db_path)

        #  print(merged_db_path)
        merged_db = self._getSchDBWithPath(merged_db_path, properties)
        second_db = self._getSchDBWithPath(second_db_path, properties)

        property_list = []
        atoms_list = []
        name_list = []
        #  for i in tqdm.trange(len(second_db)):
        #      file_base = second_db.get_name(i)
        #      property_values = []
        #      for propert in properties:
        #          mol = second_db.get_atoms(i)
        #          target_propert = second_db[i][propert]
        #          target_propert = np.array(target_propert, dtype=np.float32)
        #          property_values.append(target_propert)

        #      # combine two lists into a dictionary
        #      property_dict = dict(zip(properties, property_values))
        #      atoms_list.append(mol)
        #      name_list.append(file_base)
        #      property_list.append(property_dict)

        idxs = range(len(second_db))
        # implementation of  multiprocessor in tqdm.
        # Ref.https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
        pool = Pool(processes=num_processes)
        for result in tqdm.tqdm(
            pool.imap_unordered(func=self._mergeDataBases, iterable=idxs),
            total=len(idxs)):
            if result:
                atoms_list.append(result[0])
                name_list.append(result[1])
                property_list.append(result[2])
        merged_db.add_systems(atoms_list, name_list, property_list)

    def printDBSpec(self, idx):
        self._loadSchDB()
        print('Number of reference calculations:', len(self.db))
        print('Available properties:')
        for p in self.db.available_properties:
            print(p)

        #  file_base = self.db.get_name(idx)
        #  print("Spec of %d. Molecule %s \n" % (idx, file_base), self.db[idx])
        #  print
        #  i = 0
        #  example = db[i]
        #  print('Properties of molecule with id %s:' % i)
        #  for k, v in example.items():
        #      print('-', k, ':', v.shape)

    def optainCoordFile(self, db_path, idx):
        from ase.io import write
        self._loadSchDB()
        file_base = self.db.get_name(idx)
        atoms = self.db.get_atoms(idx)
        write("%s.xyz" % file_base, atoms)

    def _getEneryForcesFromAseDB(self, file_base):
        self._loadAseDB()
        for row in self.db:
            if file_base == row.name:
                return row.energy, row.forces

    def _hartree2Unit(self, i):
        #  HARTREE2KCAL = 627.5094738898777
        HARTREE2KCAL = units.Hartree / (units.kcal / units._Nav)

        if self.UNIT == "kcal":
            CONVERTER = HARTREE2KCAL

        file_base = self.db.get_name(i)
        mol = self.db.get_atoms(i)
        #  properties = ["energy", "forces", "dipole_moment"]

        #  for propert in properties:
        # NOTE: Due to precision lose, instead of np.float32  we use np.float64
        # We also convert tensor to python float64 firstly..
        #  energy = np.array(np.float32(self.db[i][properties[0]].item()) * HARTREE2KCAL, dtype=np.float32)
        if self.UNIT == "ev":
            #we use nature ase energy and forces in eV unit
            energy, forces = self._getEneryForcesFromAseDB(file_base)
            energy = np.array(energy, dtype=np.float)
            forces = np.array(forces, dtype=np.float)
            # load schDB for exract dipole moment from schDB
            self._loadSchDB()
        else:
            energy = np.array(self.db[i][properties[0]].item() * CONVERTER, dtype=np.float)
            forces = np.array((self.db[i][properties[1]]) * CONVERTER / units.Bohr, dtype=np.float)

        dipole_moment = np.array(self.db[i][properties[2]], dtype=np.float)

        # combine two lists into a dictionary
        property_dict = dict(zip(properties, [energy, forces, dipole_moment]))
        return mol, file_base, property_dict

    def hartree2UnitDB(self, num_processes, new_db_path, UNIT):


        self._loadSchDB()
        if os.path.exists(new_db_path):
            os.remove(new_db_path)

        self._setUnit(UNIT)

        #  properties = ["energy", "forces", "dipole_moment"]
        new_db = AtomsData(new_db_path,
                           available_properties=properties)
        lenDB = len(self.db)
        property_list = []
        atoms_list = []
        name_list = []

        # implementation of  multiprocessor in tqdm.
        # Ref.https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
        pool = Pool(processes=num_processes)
        for result in tqdm.tqdm(
            pool.imap_unordered(func=self._hartree2Unit,
                                iterable=range(lenDB)), total=lenDB):
            if result:
                atoms_list.append(result[0])
                name_list.append(result[1])
                property_list.append(result[2])
        new_db.add_systems(atoms_list, name_list, property_list)


parser = argparse.ArgumentParser(description="")
parser.add_argument("-calcMode", "--calcMode", type=str, required=False,
                    help="enter target function name exactly")
parser.add_argument("-dbPath", "--dbPath", type=str, required=False)

args = parser.parse_args()

properties = ["energy", "forces"]
mof_num = "5"
BASE_DIR = "/truba_scratch/yzorlu/deepMOF/HDNNP/prepare_data"
#  dbName = "nonEquGeometriesEnergyForecesDMomentWithORCA_TZVP_fromScalingIRMOFseries%s_ev.db" %mof_num
dbName = "nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling_IRMOFseries1_4_6_7_10_merged_50000_ev.db"

# False test datbase
#  dbName = "nonEquGeometriesEnergyForecesDMomentWithORCA_TZVP_fromScalingIRMOFseries%s_ev_testData.db" %mof_num

if args.calcMode == "mergeDataBases":
    db_path = "%s/workingOnDataBase/nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling.db" %BASE_DIR
elif args.calcMode == "print_data" or args.calcMode == "calculatedFiles2csv" or args.calcMode == "energiesFmax2csv":
    db_path = args.dbPath
else:
    db_path = "%s/workingOnDataBase/%s" %(BASE_DIR, dbName)

getprint = GetPrintDB(db_path, BASE_DIR)

if args.calcMode == "calculatedFiles2csv":
    #  out_csv_path = "%s/dataBases/IRMOFseries%s_CalculatedFilesFrom_outOfSFGeomsTZVP.csv" %(BASE_DIR, mof_num)
    out_csv_path = db_path.replace(".db", ".csv")
    getprint.calculatedFiles2csv(out_csv_path, frag_base=None)

if args.calcMode == "mergeDataBases":
    # if not equal number of properties between main db and second db
    # you must to assing fewer prorpeties db to main db (self.db)
    second_db_path = "%s/dataBases/nonEquGeometriesEnergyForecesDMomentWithORCA_TZVP_fromScalingIRMOFseries%s_ev.db" %(BASE_DIR, mof_num)
    merged_db_path = "%s/workingOnDataBase/nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling_IRMOFseries%s_merged_50000_ev.db" %(BASE_DIR, mof_num)
    getprint.mergeDataBases(100, second_db_path, merged_db_path)

if args.calcMode == "partOfDB2NewDB":
    new_db_path = "%s/dataBases/nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling_v0.db" % BASE_DIR
    getprint.partOfDB2NewDB(new_db_path)

if args.calcMode == "energiesFmax2csv":
    out_csv_path = "./energiesFmax_irmofseries%s_ev.csv" %mof_num
    getprint.energiesFmax2csv(num_processes=100, out_csv_path=out_csv_path)

if args.calcMode == "selectedDB2DB":
    new_db_path = "%s/workingOnDataBase/selected_%s" %(BASE_DIR, dbName)
    getprint.selectedDB2DB(num_processes=100, new_db_path=new_db_path)

if args.calcMode == "print_data":
    print("mof_num: ", mof_num)
    getprint.printDBSpec(0)

if args.calcMode == "hartree2UnitDB":
    UNIT = "ev"
    new_db_path = "%s/workingOnDataBase/TESTnonEquGeometriesEnergyForecesDMomentWithORCA_TZVP_fromScalingIRMOFseries_%s_v2.db" % (BASE_DIR, UNIT)
    getprint.hartree2UnitDB(num_processes=100, new_db_path=new_db_path, UNIT=UNIT)

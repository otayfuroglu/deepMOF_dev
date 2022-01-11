#
from ase.io import read
from ase.db import connect

from schnetpack import AtomsData
import os
#  from dftd4 import D4_model
from ase.calculators.orca import ORCA
#from gpaw import GPAW, PW

import numpy as np
import pandas as pd
import multiprocessing
from .orca_parser import OrcaParser



def orca_calculator(label, n_cpu, initial_gbw=['', '']):
    return ORCA(label=label,
                maxiter=40,
                charge=0, mult=1,
                orcasimpleinput='SP PBE D4 DEF2-TZVP DEF2/J RIJDX MINIPRINT NOPRINTMOS NOPOP NoKeepInts NOKEEPDENS ' + initial_gbw[0],
                orcablocks='%scf Convergence normal \n maxiter 40 end \n %pal nprocs ' + str(n_cpu) + ' end' + initial_gbw[1]
                )


class CaculateData():
    def __init__(self, properties, fragBase, nFragments, nMolecules, n_cpu,
                 filesDIR, db_path, db_name, csv_name, UNIT,
                ):
        self.filesDIR = filesDIR
        self.db_path = db_path
        self.db_name = db_name
        self.csv_name = csv_name
        self.i = 0

        self.properties = properties

        # create new data base
        self.new_db = None
        self._createNewDb()

        self.fragBase = fragBase
        self.nFragments = nFragments
        self.nMolecules = nMolecules
        self.n_cpu = n_cpu

        # laod files
        self.file_names = None
        self._getFileNames()

        # load fragments
        self.fragmentFiles = None
        self._getFragments()

        self._checkCSVFile()

        self.UNIT = "ev"

        self.rm_files = True

    def _setOrcaParser(self):
        self.fromOrcaParser = True

    def rmNotConvFiles(self):
        self.rm_files = True

    def _getFileNames(self):
        self.file_names = os.listdir(self.filesDIR)

    def _getFragments(self):
        fragments = []
        for i in range(1, self.nMolecules+1):
            for j in range(1, self.nFragments+1):
                fragments.append([item for item in self.file_names if "%s%d_f%d"%(self.fragBase, i, j) in item])
        # drop empty item in fragment list
        fragments = list(filter(None, fragments))
        self.fragmentFiles = fragments

    def _checkCSVFile(self):
        if not os.path.exists("%s/%s" %(self.db_path, self.csv_name)):
            df = pd.DataFrame(columns=["FileNames"])
            df.to_csv("%s/%s" %(self.db_path, self.csv_name), index=None)

    def _createNewDb(self):
        self.new_db = AtomsData("%s/%s" %(self.db_path, self.db_name), available_properties=self.properties)

    def _add_calculated_file(self, df_calculated_files, file_base):
        df_calculated_files_new = pd.DataFrame([file_base], columns=["FileNames"])
        df_calculated_files_new.to_csv("%s/%s" %(self.db_path, self.csv_name), mode='a', header=False, index=None)

    def _calculate_data(self, file_name):
        file_base = file_name.split(".")[0]
        initial_gbw_name = "initial_" + file_base.split("_")[0]\
                + "_" + file_base.split("_")[1] + ".gbw"

        # db_calculated = connect("%s/%s" %(self.db_path, self.db_name)).select()
        # calculated_files = [row["name"] for row in db_calculated]
        df_calculated_files = pd.read_csv("%s/%s" %(self.db_path, self.csv_name), index_col=None)
        calculated_files = df_calculated_files["FileNames"].to_list()
        if file_base in calculated_files:
            #  print("The %s file have already calculated" %file_base)
            self.i += 1
            return None

        # Fistyl, file base will be add to calculted csv file
        self._add_calculated_file(df_calculated_files, file_base)
        mol = read("%s/%s" %(self.filesDIR, file_name))

        label = "orca_%s" %file_base
        temp_files = os.listdir(os.getcwd())
        try:
            if initial_gbw_name in temp_files:
                initial_gbw = ['MORead',  '\n%moinp "{}"'.format(initial_gbw_name)]
                mol.set_calculator(orca_calculator(label, self.n_cpu, initial_gbw))
            else:
                mol.set_calculator(orca_calculator(label, self.n_cpu))

            if self.UNIT == "au":
                mol.get_potential_energy()

                datafile = label+".out"
                #  orca_parser = OrcaParser(self.properties)
                orca_parser = OrcaParser(["energy", "forces", "dipole_moment"])
                all_properties = orca_parser.getProperties(datafile)

                # parameters are list of properties dict for add_systems
                self.new_db.add_systems([mol], ["%s"%file_base], [all_properties])
            if self.UNIT == "ev":
                energy = mol.get_potential_energy()
                forces = mol.get_forces()

                energy = np.array([energy], dtype=np.float)
                forces = np.array([forces], dtype=np.float)

                datafile = label+".out"
                orca_parser = OrcaParser(["dipole_moment"])
                dipol_moment = orca_parser.getProperties(datafile)

                all_properties = {}
                all_properties["energy"] = energy
                all_properties["forces"] = forces

                # append dipol_moment dict to all_properties dict
                all_properties.update(dipol_moment)

                # parameters are list of properties dict for add_systems
                self.new_db.add_systems([mol], ["%s"%file_base], [all_properties])
            else:
                print("Plase set unit as ev or au!!!")
                exit(1)

            if self.i == 0:
                os.system("mv %s.gbw %s" %(label, initial_gbw_name))

            os.system("rm %s*" %label)
            self.i += 1
        except:
            #  print("Error for %s" %file_base)
            self.i += 1

            # remove this non SCF converged file from xyz directory.
            if self.rm_files:
                os.remove("%s/%s" %(self.filesDIR, file_name))
                #  print(file_name, "Removed!")

            # remove all orca temp out files related to label from runGeom directory.
            os.system("rm %s*" %label)

    def calculate_data(self, n_proc):
        #atoms_list = []
        #property_list = []
        if self.fragmentFiles is None:
            print("There aren't fragment files")
            exit(1)

        for file_names in self.fragmentFiles:
            self.i = 0
            with multiprocessing.Pool(n_proc) as pool:
                pool.map(self._calculate_data, file_names)
            pool.close()


    def countFiles(self):
        return len(self.file_names)

    def checkStatus():
        pass


    def print_data(self):
        self.new_db = AtomsData("%s/%s" %(self.db_path, self.db_name))
        print('Number of reference calculations:', len(self.new_db))
        print('Available properties:')
        for p in self.new_db.available_properties:
            print(p)
        print
        i = 1
        example = self.new_db[i]
        print('Properties of molecule with id %s:' % i)
        for k, v in example.items():
            print('-', k, ':', v.shape)


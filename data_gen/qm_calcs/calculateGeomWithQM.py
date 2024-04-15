#
from ase.io import read, write
import os
#  from dftd4 import D4_model
from ase.calculators.orca import ORCA
#from gpaw import GPAW, PW

#  import numpy as np
import pandas as pd
import multiprocessing
from orca_parser import OrcaParser



def orca_calculator(label, n_task, initial_gbw=['', '']):
    return ORCA(label=label,
                maxiter=250,
                charge=0, mult=1,
                orcasimpleinput='SP PBE D4 DEF2-TZVP DEF2/J RIJDX MINIPRINT NOPRINTMOS NOPOP NoKeepInts NOKEEPDENS ' + initial_gbw[0],
                orcablocks='%scf Convergence tight \n maxiter 250 end \n %pal nprocs ' + str(n_task) + ' end' + initial_gbw[1]
                )


class CaculateData():
    def __init__(self, properties, n_task, in_extxyz, out_extxyz, csv_name):

        self.in_extxyz = in_extxyz
        self.out_extxyz = out_extxyz
        self.csv_name = csv_name
        #  self.i = 0

        # for initial gbw
        self.create_gbw = True

        self.atoms_list = None
        self._loadAtaoms()
        self.properties = properties

        self.n_task = n_task

        self._checkCSVFile()

        self.rm_files = False

    def _loadAtaoms(self):
        self.atoms_list = read(self.in_extxyz, index=":")

    def _setOrcaParser(self):
        self.fromOrcaParser = True

    def rmNotConvFiles(self):
        self.rm_files = True

    def _checkCSVFile(self):
        if not os.path.exists(self.csv_name):
            df = pd.DataFrame(columns=["FileNames"])
            df.to_csv(self.csv_name, index=None)

    def _add_calculated_file(self, df_calculated_files, file_base):
        df_calculated_files_new = pd.DataFrame([file_base], columns=["FileNames"])
        df_calculated_files_new.to_csv(self.csv_name, mode='a', header=False, index=None)

    def _calculate_data(self, idx):
        atoms = self.atoms_list[idx]
        file_base = atoms.info["label"]
        initial_gbw_name = "initial_" + file_base.split("_")[0] + ".gbw"

        df_calculated_files = pd.read_csv(self.csv_name, index_col=None)
        calculated_files = df_calculated_files["FileNames"].to_list()
        if file_base in calculated_files:
            print("The %s file have already calculated" %file_base)
            #  self.i += 1
            return None

        # file base will be add to calculted csv file
        self._add_calculated_file(df_calculated_files, file_base)

        label = "orca_%s" %file_base
        temp_files = os.listdir(os.getcwd())

        try:
            if initial_gbw_name in temp_files:
                initial_gbw = ['MORead',  '\n%moinp "{}"'.format(initial_gbw_name)]
                atoms.set_calculator(orca_calculator(label, self.n_task, initial_gbw))
            else:
                atoms.set_calculator(orca_calculator(label, self.n_task))

            # orca calculation start
            atoms.get_potential_energy()

            #  if self.i == 0:
            if self.create_gbw:
                os.system("mv %s.gbw %s" %(label, initial_gbw_name))
                self.create_gbw = False

            write(self.out_extxyz, atoms, append=True)
            os.system("rm %s*" %label)
            #  self.i += 1
        except:
            #  print("Error for %s" %file_base)
            #  self.i += 1

            # remove this non SCF converged file from xyz directory.
            if self.rm_files:
                os.remove("%s/%s" %(self.filesDIR, file_name))
                #  print(file_name, "Removed!")

        #      # remove all orca temp out files related to label from runGeom directory.
            os.system("rm %s*" %label)
            return None

    def countAtoms(self):
        return len(self.atoms_list)

    def calculate_data(self, n_proc):

        #  self.i = 0
        idxs = range(self.countAtoms())
        with multiprocessing.Pool(n_proc) as pool:
            pool.map(self._calculate_data, idxs)
        pool.close()



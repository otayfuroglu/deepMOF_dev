#
from ase.io import read, write
import os, shutil
#  from dftd4 import D4_model
from ase.calculators.orca import ORCA
#from gpaw import GPAW, PW

#  import numpy as np
import pandas as pd
import multiprocessing
from orca_parser import OrcaParser

from pathlib import Path
from orca_io import (read_orca_h_charges,
                     read_orca_chelpg_charges,
                     read_orca_ddec_charges,
                    )




def prepareDDECinput(label):
    input_txt = f"""
        <periodicity along A, B, and C vectors>
        .false.
        .false.
        .false.
        </periodicity along A, B, and C vectors>

        <atomic densities directory complete path>
        /arf/home/otayfuroglu/miniconda3/pkgs/chargemol-3.5-h1990efc_0/share/chargemol/atomic_densities/
        </atomic densities directory complete path>

        <input filename>
        {label}.wfx
        </input filename>

        <charge type>
        DDEC6
        </charge type>

        <compute BOs>
        .false.
        </compute BOs>
    """
    with open("job_control.txt", "w") as fl:
        print(input_txt, file=fl)


def orca_calculator(label, calc_type,  n_task, initial_gbw=['', '']):
    return ORCA(
        label=label,
        maxiter=250,
        charge=0, mult=1,
        orcasimpleinput= calc_type.upper() \
            + ' PBE D4 DEF2-TZVP DEF2/J RIJDX MINIPRINT NOPRINTMOS NoKeepInts NOKEEPDENS CHELPG AIM ' \
            + initial_gbw[0],
        orcablocks= '%scf Convergence tight \n maxiter 250 end \n %output \n' \
            + ' Print[ P_Hirshfeld] 1 end \n %pal nprocs ' \
            + str(n_task) + ' end' \
            + initial_gbw[1]
    )


class CaculateData():
    def __init__(self, properties, calc_type, n_task, in_extxyz_path, out_extxyz_path, csv_path):

        self.in_extxyz_path = in_extxyz_path
        self.out_extxyz_path = out_extxyz_path
        self.csv_path = csv_path
        #  self.i = 0

        self.atoms_list = None
        self._loadAtaoms()
        self.properties = properties

        self.calc_type = calc_type
        self.n_task = n_task

        self._checkCSVFile()

        self.rm_files = False

    def _loadAtaoms(self):
        self.atoms_list = read(self.in_extxyz_path, index=":")

    def _setOrcaParser(self):
        self.fromOrcaParser = True

    def rmNotConvFiles(self):
        self.rm_files = True

    def _checkCSVFile(self):
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=["FileNames"])
            df.to_csv(self.csv_path, index=None)

    def _add_calculated_file(self, df_calculated_files, label):
        df_calculated_files_new = pd.DataFrame([label], columns=["FileNames"])
        df_calculated_files_new.to_csv(self.csv_path, mode='a', header=False, index=None)

    def _calculate_data(self, idx):
        atoms = self.atoms_list[idx]
        try:
            label = atoms.info["label"]
        except:
            label = "frame_" + "{0:0>5}".format(idx)
            atoms.info["label"] = label


        df_calculated_files = pd.read_csv(self.csv_path, index_col=None)
        calculated_files = df_calculated_files["FileNames"].to_list()
        if label in calculated_files:
            print("The %s file have already calculated" %label)
            #  self.i += 1
            return None

        # file base will be add to calculted csv file
        self._add_calculated_file(df_calculated_files, label)

        #  label = "orca_%s" %label

        cwd = os.getcwd()

        #  full path
        OUT_DIR = Path(cwd) / Path("run_" + self.in_extxyz_path.split('/')[-1].split(".")[0]) / Path(label)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        GBW_DIR = Path(cwd) / Path("run_" + self.in_extxyz_path.split('/')[-1].split(".")[0])
        initial_gbw_name = "initial_" + label.split("_")[0] + ".gbw"
        initial_gbw_file = [flname for flname in os.listdir(GBW_DIR) if ".gbw" in flname]

        print(f"working in {OUT_DIR} directory\n")
        os.chdir(OUT_DIR)

        try:
            if len(initial_gbw_file) == 1:
                shutil.copy2(f"{GBW_DIR}/{initial_gbw_name}", OUT_DIR)
                initial_gbw = ['MORead',  '\n%moinp "{}"'.format(initial_gbw_name)]
                atoms.set_calculator(orca_calculator(label, self.calc_type, self.n_task, initial_gbw))
            else:
                atoms.set_calculator(orca_calculator(label, self.calc_type, self.n_task))

            # orca calculation start
            atoms.get_potential_energy()

            # get hirshfeld point chargess externally
            h_charges = read_orca_h_charges(f"{label}.out")
            atoms.arrays["HFPQ"] = h_charges

            chelpg_charges = read_orca_chelpg_charges(f"{label}.pc_chelpg")
            atoms.arrays["CHELPGPQ"] = chelpg_charges

            prepareDDECinput(label)
            os.system("/arf/home/otayfuroglu/miniconda3/pkgs/chargemol-3.5-h1990efc_0/bin/chargemol")

            ddec_charges = read_orca_ddec_charges("DDEC6_even_tempered_net_atomic_charges.xyz")
            atoms.arrays["DDECPQ"] = ddec_charges

            #  if self.i == 1:
            if len(initial_gbw_file) == 0:
                os.system("mv %s.gbw %s/%s" %(label, GBW_DIR, initial_gbw_name))

            #  os.system("rm %s*" %label)
            os.chdir(cwd)
            write(self.out_extxyz_path, atoms, append=True)
            shutil.rmtree(OUT_DIR)
                #  self.i += 1
        except:
            #  print("Error for %s" %label)
            #  self.i += 1

            # remove this non SCF converged file from xyz directory.
            if self.rm_files:
                os.remove("%s/%s" %(self.filesDIR, file_name))
                #  print(file_name, "Removed!")

            # remove all orca temp out files related to label from runGeom directory.
            #  os.system("rm %s/%s*" %(OUT_DIR, label))
            shutil.rmtree(OUT_DIR)
            os.chdir(cwd)
            return None

    def countAtoms(self):
        return len(self.atoms_list)

    def calculate_data(self, n_proc):

        #  self.i = 0
        idxs = range(self.countAtoms())
        with multiprocessing.Pool(n_proc) as pool:
            pool.map(self._calculate_data, idxs)
        pool.close()



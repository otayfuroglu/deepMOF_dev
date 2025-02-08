#
import RASPA

#  import openbabel
from openbabel import pybel
import pandas as pd
#  import sys, os

import multiprocessing
import multiprocessing.pool


import argparse
import os

def get_helium_void_fraction(cif_file_path):
    print("Helium void fraction is calculating...")
    structure = next(pybel.readfile("cif", cif_file_path))
    result = RASPA.get_helium_void_fraction(structure,
                                    cycles=1000,
                                    unit_cells=(1,1,1),
                                    #  forcefield="GenericMOFs",
                                    forcefield="CrystalGenerator",
                                    input_file_type="cif",
                                   )
    print("Done")
    return result


def runRASPA(cif_file_path):

    structure = next(pybel.readfile("cif", cif_file_path))

    helium_void_fraction = get_helium_void_fraction(cif_file_path)
    #  helium_void_fraction = 1.0

    print(f"RASPA is calculating for {cif_file_path}")

    results = RASPA.run(
        structure, molecule,
        simulation_type="MonteCarlo",
        temperature=77, # in Kelvin
        pressure=1e5, # in Pascal
        helium_void_fraction=helium_void_fraction,
        unit_cells=(2,2,2),
        #  unit_cells=(2,2,2),
        framework_name="streamed", # if not streaming, this will load the structure at `$RASPA_DIR/share/raspa/structures`.
        cycles=1000,
        #  init_cycles="auto",
        init_cycles=500,
        #  forcefield="GenericMOFs",
        forcefield="CrystalGenerator",
        input_file_type="cif",
    )

    print(f"Done for {cif_file_path}")

    return results


def prun(cif_file_path):

    results = runRASPA(cif_file_path)

    df["refcode"] = [f"{cif_file_path.split('/')[-1].split('.')[0]}"]
    for label in labels:
        if "Henry" in label:
            df[label] = [results[label]["Henry"][0]]
        else:
            df[label] = results["Number of molecules"][molecule][label][0]
    df.to_csv(csv_file_path, mode="a", header=False, index=False)


# test
def test():
    cif_file_path = "cif_files/filled_IRMOF1.cif"
    molecule = "H2"
    results = runRASPA(cif_file_path)
    print(results["Average Henry coefficient"]["Henry"][0])


#for correction AssertionError: daemonic processes are not allowed to have children
# tested in python3.6
class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)


if "__main__" == __name__:
    parser = argparse.ArgumentParser(description="Give something ...")
    parser.add_argument("-cifdir", type=str, required=True, help="..")
    parser.add_argument("-ncore", type=int, required=True, help="..")
    args = parser.parse_args()

    molecule = "H2"
    CIFDIR = args.cifdir
    cif_file_paths = [f"{CIFDIR}/{fl}" for fl in os.listdir(CIFDIR) if ".cif" in fl]

    # to extract desired results from RASPA output
    labels = [
        "Average loading absolute [milligram/gram framework]",
        "Average loading absolute [cm^3 (STP)/cm^3 framework]",
        "Average loading excess [milligram/gram framework]",
        "Average loading excess [cm^3 (STP)/cm^3 framework]",
        "Average loading excess [mol/kg framework]",
        "Average Henry coefficient",
    ]

    header = labels.copy()
    header.insert(0, "refcode")

    csv_file_path = "data.csv"

    df = pd.DataFrame(columns=header)
    df.to_csv(csv_file_path, index=False)

    pool = MyPool(processes=args.ncore)
    pool.map(prun, cif_file_paths)





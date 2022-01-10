
import os, sys
sys.path.insert(0, "/truba_scratch/otayfuroglu/deepMOF_dev")
from data_gen.calculateGeomWithQM import CaculateData
import multiprocessing
import getpass

USER = getpass.getuser()

UNIT = "ev"
mof_num = "1"
db_path = f"/truba_scratch/{USER}/deepMOF_dev/data_bases"

#  filesDIR = f"/truba_scratch/{USER}/deepMOF/HDNNP/prepare_data/outOfSFGeomsIRMOFs%s" % mof_num
#  db_name = "nonEquGeometriesEnergyForecesDMomentWithORCA_TZVP_fromScalingIRMOFseries%s_%s.db" %(mof_num, UNIT)
#  csv_name = "IRMOFseries%s_CalculatedFilesFrom_outOfSFGeomsTZVP.csv" % mof_num

# for calculation of test data
filesDIR = f"/truba_scratch/{USER}/deepMOF_dev/geom_files/MOF5/broken_bond"
db_name = "breken_bond_irmofseries%s_f1_%s_testData.db" %(mof_num, UNIT)
csv_name = "IRMOFseries%s_f1_broken_bonds.csv" % mof_num

os.chdir(os.getcwd())

#  if not os.path.exists("%s/%s" %(db_path, csv_name)):
#      df = pd.DataFrame(columns=["FileNames"])
#      df.to_csv("%s/%s" %(db_path, csv_name), index=None)
#
#  #os.chdir(filesDIR)
#  file_names = os.listdir(filesDIR)
#  print(len(file_names))

n_core = multiprocessing.cpu_count()

n_cpu = 8
if n_core == 24 or n_core == 48:
    n_cpu = 6
if n_core == 40 or n_core == 80:
    n_cpu = 8
if n_core == 28 or n_core == 56:
    n_cpu = 4
if n_core == 112:
    n_cpu = 16

n_proc = int(n_core / n_cpu)


properties = ["energy", "forces", "dipole_moment"]
fragBase = "irmof"
nFragments = 7
nMolecules = 16

calculate = CaculateData(properties, fragBase, nFragments, nMolecules, n_cpu,
                         filesDIR, db_path, db_name, csv_name, UNIT
                        )
print ("Nuber of out of range geomtries", calculate.countFiles())
print("QM calculations Running...")
calculate.calculate_data(n_proc)
print("DONE")
#  print("All process taken %2f minutes" %((time.time()- start) / 60.0))

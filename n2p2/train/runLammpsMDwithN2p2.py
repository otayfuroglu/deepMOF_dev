from ase.io.vasp import read_vasp_out
from pathlib import Path
import os
import shutil
import argparse

from ase.io.lammpsdata import write_lammps_data


try:
    import multiprocessing
    n_cpu = multiprocessing.cpu_count()
except:
    n_cpu = 20


def write_lammps_input():
    lammps_input = """
    ###############################################################################
    # MD simulation for NN
    ###############################################################################
    ###############################################################################
    # VARIABLES
    ###############################################################################
    clear
    variable cfgFile         string data.%s

    # NN
    variable nnpCutoff       equal  6.50
    variable nnpDir          string ./

    # variables available on command line

    variable        mu index -8.1
    variable        Pressu          equal  1
    # variable        Precoeff        equal  0.3864
    variable        Precoeff        equal  1.0
    variable	disp index 0.5
    variable        T index %d
    # variable        lbox index 45
    # variable        spacing index 15.0

    variable                R               equal   0.00198722
    variable                sysvol          equal   vol
    variable                sysmass         equal   mass(all)/6.0221367e+23
    variable                sysdensity      equal   v_sysmass/v_sysvol/1.0e-24
    variable                coulomb         equal   ecoul+elong
    variable                etotal          equal   etotal
    variable                pe              equal   pe
    variable                ke              equal   ke
    variable                evdwl           equal   evdwl
    variable                epair           equal   epair
    variable                ebond           equal   ebond
    variable                eangle          equal   eangle
    variable                edihed          equal   edihed
    variable                time            equal   step*dt+0.000001


    ###############################################################################
    # GENERAL SETUP
    ###############################################################################
    units metal
    dimension 3
    boundary p p p
    atom_style atomic
    read_data ${cfgFile}

    replicate 1 1 1

    mass 1  26.98153860
    mass 2  %.8f  # for metal_type
    mass 3  1.00794

    log             alanates_${Pressu}Bar_${T}K.log append
    ###############################################################################
    # NN
    ###############################################################################
    # pair_style nnp dir ${nnpDir} showew no showewsum 10 resetew no maxew 100 cflength 1.8897261328 cfenergy 0.0367493254 emap "1:H,2:C,3:O,4:Zn"
    pair_style nnp dir ${nnpDir}  maxew 300 cflength 1.8897261328 cfenergy 0.0367493254  emap "1:Al,2:%s,3:H"
    pair_coeff * * ${nnpCutoff}
    #


    ############
    # SETTINGS #
    ############

    # MD settings

    thermo 200
    #-------------------------------------------------------------------------------
    # Stage 0: minimization
    #
    # minimize 1.0e-8 1.0e-8 1000 10000
    #
    # minimization step with relaxing cell
    min_style       cg
    print           "MinStep,CellMinStep,AtomMinStep,FinalStep,Energy,EDiff" file beforeMD_alanates_${Pressu}Bar_${T}K_min.csv screen no
    variable        min_eval   equal 1.00e-8
    variable        prev_E     equal 50000.00
    variable        iter       loop 100000
    label           loop_min
    min_style       cg
    fix             1 all box/relax aniso 0.0 vmax 0.01
    minimize        1.0e-15 1.0e-15 10000 100000
    unfix           1
    min_style       fire
    variable        tempstp    equal $(step)
    variable        CellMinStep equal ${tempstp}
    minimize        1.0e-15 1.0e-15 10 100
    variable        AtomMinStep equal $(step)
    variable        temppe     equal $(pe)
    variable        min_E      equal abs(${prev_E}-${temppe})
    print           "${iter},${CellMinStep},${AtomMinStep},${AtomMinStep},$(pe),${min_E}" append beforeMD_alanates_${Pressu}Bar_${T}K_min.csv screen no
    if              "${min_E} < ${min_eval}" then "jump SELF break_min"
    variable        prev_E     equal ${temppe}
    next            iter
    jump            SELF loop_min
    label           break_min



    compute moltemp all temp
    compute_modify moltemp dynamic/dof yes
    compute_modify thermo_temp dynamic/dof yes

    neighbor        2.0 bin
    neigh_modify    every 1 delay 10 check yes
    velocity        all create ${T} 54654
    timestep        0.0005

    #  fix                     mynvt all nvt temp ${T} ${T} 10
    #  run 15000
    #  unfix mynvt

    reset_timestep 0
    dump 1 all atom 20 alanates_${Pressu}Bar_${T}K.lammpstrj



    fix                     mynpt all npt temp ${T} ${T} 10 iso 0.0 0.0 100.0
    fix_modify		mynpt temp moltemp


    thermo_style    custom step temp press pe ke density atoms vol

    run             20000

    """ % (file_base, 2*temp, metal_mass_pair[metal], metal)

    with  open(f"in.{file_base}", "w") as fl:
        fl.write(lammps_input)


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-calc_type", type=str, required=True)
parser.add_argument("-metal", type=str, required=True)
parser.add_argument("-temp", type=int, required=True)
parser.add_argument("-geom", type=str, required=True)
parser.add_argument("-idx", type=int, required=True)
parser.add_argument("-tmpdir", type=str, required=True)
parser.add_argument("-memory", type=int, required=True)
args = parser.parse_args()

#  init_geom = "optimized"
#  calc_type = "md"
#  metal = "Li"
#  temp = 200
metal_mass_pair = {"Li": 6.94000000}
calc_type = args.calc_type
metal = args.metal
temp = args.temp
init_geom = args.geom
memory = args.memory

BASE_DIR = "/kernph/tayfur0000/works/alanates"
GEOMS_DIR = f"{BASE_DIR}/Top20"
OPT_GEOMS_DIR = f"{BASE_DIR}/vasp_opt"
WORKS_DIR = f"{BASE_DIR}/n2p2/runMD/lammps_{calc_type}_{temp}K"

MODEL_DIR = f"{BASE_DIR}/n2p2/runTrain/zeta_12416_frate10_24_task_14kdata/best_epoch_90"

OUT_DIRS = []
geoms_path = []
for _DIR in [item for item in os.listdir(GEOMS_DIR) if metal in item]:
    for struct_type in ["polymeric", "isolated"]:
    #  for struct_type in ["isolated"]:
        GEOM_DIR = f"{GEOMS_DIR}/{_DIR}/{struct_type}"
        for fl_name in os.listdir(GEOM_DIR):
            fl_base = fl_name.replace('.ascii', '')
            OUT_DIRS += [f"{WORKS_DIR}/{_DIR}/{struct_type}/{fl_base}"]
            if init_geom == "optimized":
                OPT_GEOM_DIR = f"{OPT_GEOMS_DIR}/{_DIR}/{struct_type}/{fl_base}"
                geoms_path += [f"{OPT_GEOM_DIR}/OUTCAR"]
            else:
                geoms_path += [f"{GEOM_DIR}/{fl_name}"]

idx = args.idx

if idx > (len(OUT_DIRS) - 1):
        quit()

OUT_DIR = OUT_DIRS[idx]
TMP_DIR = args.tmpdir
geom_path = geoms_path[idx]

if not os.path.exists(OUT_DIR):
        #  CURRENT_DIR = os.getcwd()
        if init_geom == "optimized":
            #  atoms = read_vasp_out(f"{OPT_GEOM_DIR}/OUTCAR")
            atoms = read_vasp_out(geom_path)
        else:
            atoms = read_v_sim(geom_path)
        atoms.pbc = [1, 1, 1]

        # due to partion of memdifferent size
        # job submit according to mem size for protect out of mem
        #  if memory == 128:
        #      if len(atoms) < 20:
        #          quit()
        #  if  memory == 64:
        #      if len(atoms) >= 20:
        #          quit()
        #  elif memory == 32 and len(atoms) > 12:
        #      quit()

        # create outpur folders
        # easy create nested forder wether it is exists
        Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        # change to local scratch directory
        os.chdir(TMP_DIR)
        #  os.chdir(OUT_DIR)


        file_base = OUT_DIR.split('/')[-1]
        write_lammps_input()
        write_lammps_data(f"data.{file_base}", atoms, specorder=["Al", "Li", "H"])
        for fl in os.listdir(MODEL_DIR):
            shutil.copy(f"{MODEL_DIR}/{fl}", TMP_DIR)
            #  shutil.copy(f"{MODEL_DIR}/{fl}", OUT_DIR)

        os.system(f"mpirun -np {n_cpu} /kernph/tayfur0000/n2p2/bin/lmp_mpi -in in.{file_base} > log.out")
        for fl in os.listdir(TMP_DIR):
            shutil.copy(f"{TMP_DIR}/{fl}", OUT_DIR)

        #  calc = Vasp()
        #  if calc_type == "sp":
        #      nsw = 0
        #      setVaspCalculator(calc)
        #  if calc_type == "opt":
        #      nsw = 500
        #      setVaspCalculator(calc)
        #  elif calc_type == "md":
        #      setVaspMDCalculator(calc)
        #  atoms.calc = calc

        #  #  try:
        #  atoms.get_potential_energy()


        #  finally:
        #      for fl in os.listdir(TMP_DIR):
        #          shutil.copy(f"{TMP_DIR}/{fl}", OUT_DIR)
        #  os.chdir(CURRENT_DIR)

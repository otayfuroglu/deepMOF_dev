#
import os
import argparse
from multiprocessing import Pool
from multiprocessing import current_process
from itertools import product, repeat
from itertools import chain

def creat_input(base_name, fragType, gas, pressure, T):
    lammps_input = """

                    #
                    ###############################################################################
                    # VARIABLES
                    ###############################################################################
                    clear

                    # Masses

                    variable fragType      string  %s
                    variable gas           string  %s

                    variable cfgFile         string "data.${fragType}"

                    # variables available on command line

                    variable        mu index -8.1
                    variable        Pressu          equal %f
                    # variable        Precoeff        equal  0.3864
                    variable        Precoeff        equal  1.0
                    variable	disp index 0.5
                    variable        T index %d
                    # variable  	T_init equal ${T}*(2/3)
                    variable  	T_init equal ${T}*0.96
                    #
                    ##################
                    # INITIALIZATION #
                    ##################
                    units           real
                    atom_style      full
                    dimension 		3
                    newton 			on
                    boundary        p p p
                    pair_style      lj/cut/coul/long 12.5
                    bond_style      harmonic
                    angle_style     hybrid fourier cosine/periodic harmonic
                    dihedral_style  harmonic
                    improper_style  fourier
                    kspace_style    ewald 1e-06
                    pair_modify     shift yes tail no mix arithmetic #shifted, tail correction no, lorentz bertholt mixing rule
                    special_bonds   lj/coul 0.0 0.0 1.0
                    dielectric      1.0
                    box tilt        large

                    read_data	${cfgFile} & 		
                        		extra/atom/types 2 &
                        		extra/bond/types 1 &
                        		extra/angle/types 1 
                    
                    molecule        co2mol CO2.txt
                    
                    replicate 1 1 1
                    
                    # mass   1    24.305000000 # Mg6
                    # mass   2    12.010700000 # C_R
                    # mass   3     1.007940000 # H_
                    # mass   4    15.999400000 # O_3
                    # mass   5    15.999400000 # O_R
                    mass 		6	12.0107
                    mass 		7	15.9994 
                    
                    pair_coeff 1  6  0.07717237797669027 2.745705 # Mg-C_CO2
                    pair_coeff 2  6  0.0567382504870374 2.959075 # O-C_CO2
                    pair_coeff 3  6  0.0567382504870374 2.959075 # O-C_CO2
                    pair_coeff 4  6  0.07505754308078902 3.115425 # C-C_CO2
                    pair_coeff 5  6  0.04858775472001682 2.685565 # H-C_CO2
                    
                    pair_coeff 1  7  0.13200596191574038 2.870705 # Mg-O_CO2
                    pair_coeff 2  7  0.0970526959169234 3.084075 # O-O_CO2
                    pair_coeff 3  7  0.0970526959169234 3.084075 # O-O_CO2
                    pair_coeff 4  7  0.12838846531856746 3.240425 # C-O_CO2
                    pair_coeff 5  7  0.08311099731926326 2.810565 # H-O_CO2
                    
                    pair_coeff 6  6  0.0536541755 2.8 # C_CO2-C_CO2
                    pair_coeff 7  6  0.09177728137148701 2.925 # O_CO2-C_CO2
                    pair_coeff 7  7  0.156988143 3.05 # O_CO2-O_CO2

                    bond_coeff      9       0       1.16
                    angle_coeff     17  harmonic 0 	180
                    
                    # GROUPS
                    group		co2 type 6 7 #Group definition of co2 molecule
                    group           fram  type  1 2 3 4 5  #Group definition of Framework
                    
                    ############
                    # SETTINGS #
                    ############
                    reset_timestep  0 
                    neighbor 		2.0 bin
                    neigh_modify 	every 1 delay 10 check yes
                    
                    #min_style 		cg
                    #minimize		0.0 1e-08 100000 10000000
                    #run 			0
                    
                    variable 		nAds equal count(co2)/3
                    variable 		dt equal dt
                    variable  		tdamp equal ${dt}*100
                    variable		pdamp equal ${dt}*1000
                    compute 		mdtemp co2 temp
                    compute_modify 	mdtemp dynamic/dof yes
                    #fix 			1 all npt temp 313.15 313.15 ${tdamp} iso 45 45 ${pdamp}
                    velocity 		all create ${T_init} 2435728
                    #thermo 			500
                    #timestep 		1
                    #run 			10000
                    #unfix 			1
                    #write_data 		dataAfterNPT.data

                    variable        tfac equal 5.0/3.0 # (3 trans + 2 rot)/(3 trans)
                    
                    fix 			2 co2 gcmc 50 100 100 0 342134 ${T} ${mu} ${disp} pressure ${Pressu} mol co2mol tfac_insert ${tfac} group co2
                    #  fix 			3 all nvt temp ${T} ${T}  ${tdamp}
                    thermo 			100
                    thermo_style 	custom step v_nAds press temp f_2[3] f_2[4] f_2[5] f_2[6] 
                    
                    # log             ${fragType}_${gas}_${Pressu}Bar_${T}K.log append
                    log             ${fragType}_${gas}_${Pressu}Bar_${T}K.log
                    dump 1 all atom 20 ${fragType}_${gas}_${Pressu}Bar_${T}K.lammpstrj
                    # dump_modify 1  pbc yes
                    
                    timestep 		1
                    run 			50000
                    
                """ % (fragType, gas, pressure, T)



    fl = open(f"in.{base_name}", "w")
    print(lammps_input, file=fl)


def run(idx, pressure):
    adjust_p = 0
    idx = idx[0]
    pressure = pressure[idx]
    T = 298
    gas = "CO2"
    fragType = "mgmof74_exp_charged_chelpg_initial_1_1_1_P1"
    base_name = f"{fragType}_{pressure}_{T}"
    creat_input(base_name, fragType, gas, pressure, T)
    os.system(f"mpirun --cpu-set {int(idx) + adjust_p} -n 1 /truba_scratch/otayfuroglu/deepMOF_dev//nequip/lammps_stress/lammps/build/lmp -in in.{base_name} > log_{pressure}_{T}.log 2>&1")


#  parser = argparse.ArgumentParser(description="")
#  parser.add_argument("-pressure", type=float, required=True)
#  parser.add_argument("-nproc", type=int, required=True)

#  args = parser.parse_args()

#  pressures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
pressures = [p/100 for p in chain(range(1, 11, 1), range(10, 101, 10))]

n_pressures = len(pressures)
with Pool(n_pressures) as pool:
    pool.starmap(run, zip(product(range(n_pressures), repeat=1),
                          repeat(pressures)))

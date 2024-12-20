#!/bin/bash
#
SCRIPT_DIR=/arf/scratch/otayfuroglu/deepMOF_dev/gcmc/src/
model_path="/arf/scratch/otayfuroglu/deepMOF_dev/nequip/works/mof74/runTrain/results/MgF2_nnp1/nonbonded_CO2_v10_e10_p1000_r6_v3/MgF2_nonbonded_v10_nnp1_e10_p1000_r6.pth"
struc_path="../../../MgMOF74_clean_fromCORE.cif"
molecule_path="../../../co2.xyz"
T=298

# for P in 0.04 0.06 0.08 0.1
# for P in 0.2 0.4 0.6 0.8
# for P in 1 5 10 20 30 40 50
# for P in 0.001 0.02 0.04 0.06 0.08 0.1
# for P in 0.2 0.4 0.6 0.8 1.0
# for P in 0.2 0.4 0.6 0.8
P=0.1
stepsize=1.0
for mdsteps in 100
do
	$HOME/miniconda3/envs/nequip_old/bin/python $SCRIPT_DIR/runGCMCMD.py\
	       	-pressure $P\
	       	-temperature $T\
	       	-stepsize $stepsize\
	       	-totalsteps 1000000\
	       	-nmdsteps $mdsteps\
	       	-ngcmcsteps 250\
	       	-nmcmoves 250\
	       	-flex_ads no\
	       	-opt no\
		-interval 50\
	       	-model_gcmc_path $model_path\
	       	-model_md_path $model_path\
	       	-struc_path $struc_path\
	       	-molecule_path $molecule_path
done
wait # wait for the last jobs to finish

exit

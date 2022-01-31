#! /usr/bin/bash

SCRIPTDIR="/truba_scratch/yzorlu/deepMOF/HDNNP/schnetpack"

MODEL_DIR="schnet_l3_basis96_filter64_interact3_gaussian20_rho001_lr00001_bs1_cutoff_60_withoutStress_aseEnv_IRMOFseries1_4_6_7_10_merged_173014_ev"

for mof_num in  1 4 6 7 10
do
	for val_type in test train
	do
		echo $val_type to IRMOF$mof_num 

		RESULT_DIR="/truba_scratch/yzorlu/deepMOF/HDNNP/schnetpack/runTest/IRMOF$mof_num/$MODEL_DIR"
		# if not exists create
		mkdir -p $RESULT_DIR

		$SCRIPTDIR/calc_SP_with_models.py \
			-val_type $val_type -mof_num $mof_num -MODEL_DIR $MODEL_DIR -RESULT_DIR $RESULT_DIR 
		done
	done
echo All done
















#! /usr/bin/bash

SCRIPTDIR="/truba_scratch/yzorlu/deepMOF/HDNNP/schnetpack"

MODEL_DIR="schnet_l3_basis96_filter64_interact3_gaussian20_rho001_lr00001_bs1_cutoff_60_withoutStress_aseEnv_IRMOFseries1_4_6_7_10_merged_173014_ev"

for val_type in xyz_files #aliphatic_ch_bond # aromatic_ch_bond torsion
do
	echo $val_type

	RESULT_DIR="/truba_scratch/yzorlu/deepMOF/HDNNP/schnetpack/runTest/optimized_IRMOF7_linker_torsion36x36"
	# if not exists create
	mkdir -p $RESULT_DIR

	$SCRIPTDIR/calc_SP_with_models.py \
		-val_type $val_type -MODEL_DIR $MODEL_DIR -RESULT_DIR $RESULT_DIR 
done
echo All done

































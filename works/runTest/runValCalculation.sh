#! /usr/bin/bash

SCRATCH=/truba_scratch/$USER
SCRIPTDIR=$SCRATCH/deepMOF_dev/prediction
PYDIR=$HOME/miniconda3/envs/python38/bin

MODEL_DIR="schnet_l3_basis96_filter64_interact3_gaussian20_rho001_lr00001_bs1_cutoff_60_withoutStress_aseEnv_IRMOFseries1_4_6_7_10_merged_173014_ev"

filesDIR=$SCRATCH/deepMOF/HDNNP/prepare_data/geomFiles/MOF5/broken_bond
for val_type in fromFiles #aliphatic_ch_bond # aromatic_ch_bond torsion
do
	echo $val_type
	RESULT_DIR=$SCRATCH/deepMOF_dev/works/runTest/broken_bond
	# if not exists create
	mkdir -p $RESULT_DIR

	$PYDIR/python $SCRIPTDIR/calcSPwithSchnetpack.py \
		-val_type $val_type -MODEL_DIR $MODEL_DIR -RESULT_DIR $RESULT_DIR -filesDIR $filesDIR
done
echo All done

































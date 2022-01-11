#! /usr/bin/bash

SCRATCH=/truba_scratch/$USER
SCRIPTDIR=$SCRATCH/deepMOF_dev/prediction
PYDIR=$HOME/miniconda3/envs/python38/bin

MODEL_DIR="schnet_l3_basis96_filter64_interact3_gaussian20_rho001_lr00001_bs1_cutoff_60_withoutStress_aseEnv_IRMOFseries1_4_6_7_10_merged_173014_ev"

filesDIR=$SCRATCH/deepMOF/HDNNP/prepare_data/geomFiles/MOF5/broken_bond
db_path=$SCRATCH/deepMOF_dev/data_bases/breken_bond_irmofseries1_f1_ev_testData.db

for val_type in fromDB
do
	echo $val_type
	RESULT_DIR=$SCRATCH/deepMOF_dev/works/runTest/broken_bond
	# if not exists create
	mkdir -p $RESULT_DIR

	$PYDIR/python $SCRIPTDIR/calcSPwithSchnetpack.py \
		-val_type $val_type -MODEL_DIR $MODEL_DIR \
		-RESULT_DIR $RESULT_DIR -filesDIR $filesDIR \
		-db_path $db_path
done
echo All done

































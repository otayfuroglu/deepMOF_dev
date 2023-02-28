#! /usr/bin/bash

# data_path="/truba_scratch/otayfuroglu/deepMOF/HDNNP/prepare_data/dataBases/nonEquGeometriesEnergyForcesWithORCAFromMD_testSet.db"
data_path="/truba_scratch/otayfuroglu/deepMOF/HDNNP/prepare_data/dataBases/nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling.db"
BASE_DIR="/truba_scratch/otayfuroglu/deepMOF_dev/n2p2"
SCRIPTDIR="$BASE_DIR/prediction"
MODEL_DIR="$BASE_DIR/works/runTrain/weighted_rho5_zeta16_r20_l2n30_scalinkMaxMinSF_subRefE_shift_center_gastegger_core24_alldata/"

for val_type in test
do

	# RESULT_DIR="/truba_scratch/otayfuroglu/deepMOF/HDNNP/schnetpack/runTest/$MODEL_DIR"
	RESULT_DIR="$BASE_DIR/works/runTest/batch24_alldata"
	# if not exists create
	mkdir -p $RESULT_DIR

	# python $SCRIPTDIR/calc_SP_with_models.py \
	python $SCRIPTDIR/runCalcWithN2p2.py \
		-val_type $val_type -data_path $data_path -MODEL_DIR $MODEL_DIR -RESULT_DIR $RESULT_DIR 
done
echo All done






















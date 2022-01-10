#! /usr/bin/bash

NSAMPLE=20000
MDTYPE="classic"
MODE="active"
# MODE="ensembleActive"
mof_num=10
len_data=58754
nSamplesPerFragment=400
threshold=0.12

BASEDIR="/truba_scratch/yzorlu/deepMOF/HDNNP"
PYTHON38DIR="/truba/home/yzorlu/miniconda3/bin/"
# PYTHONDIR="/truba/home/yzorlu/miniconda3/bin/"
SCRIPTDIR="/truba_scratch/yzorlu/deepMOF/HDNNP/schnetpack"
# MODEL1_DIR=$BASEDIR/schnetpack/runTraining/hdnnBehler_l3n100_rho001_r20a5_lr0001_bs1_IRMOFseries${mof_num}_merged_${len_data}_ev_withStress
# MODEL2_DIR=$BASEDIR/schnetpack/runTraining/hdnnBehler_l3n50_rho001_r20a5_lr0001_bs1_IRMOFseries${mof_num}_merged_${len_data}_ev_withStress
MODEL1_DIR=$BASEDIR/schnetpack/runTraining/schnet_l3_basis96_filter64_interact3_gaussian20_rho001_lr00001_bs1_cutoff_60_withoutStress_aseEnv_IRMOFseries${mof_num}_merged_${len_data}_ev
MODEL2_DIR=$BASEDIR/schnetpack/runTraining/schnet_l4_basis96_filter64_interact3_gaussian20_rho001_lr00001_bs1_cutoff_60_withoutStress_aseEnv_IRMOFseries${mof_num}_merged_${len_data}_ev

dbBase="nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling_IRMOFseries${mof_num}_merged_50000_ev"

outOfGeomsDIR=$BASEDIR/prepare_data/outOfSFGeomsIRMOFs${mof_num}/
# for generate test data
# outOfGeomsDIR=$BASEDIR/prepare_data/testDataGeomsIRMOFs${mof_num}/
GPUIDS=(0 1) # for barbuncuda
# GPUIDS=(0 1 2 3) # for akyacuda
for k in ${mof_num}
do
	# NOTE: same fragments are missing i.e. 5 in 1 
	fragBase="irmofseries${k}_f"
	for i in {6..6}
	do
		fragName=${fragBase}${i}
		echo $fragName
		$PYTHON38DIR/python -u $SCRIPTDIR/calculateSF.py -fragName $fragName\
			-dbBase $dbBase
		
		maxMinSFPath=$PWD/$fragName"_MinMaxValues.csv"
		
		j=0
		for TEMP in {50..600..50}
		do
				CUDA_VISIBLE_DEVICES=${GPUIDS[$(((j)%${#GPUIDS[@]}))]} # to assing job each GPU
				$PYTHON38DIR/python -u $SCRIPTDIR/run_md_withSchnetpack_active.py \
					-mdtype $MDTYPE\
					-n $NSAMPLE -temp $TEMP -fragName $fragName\
					-maxMinSFPath $maxMinSFPath\
					-outOfGeomsDIR $outOfGeomsDIR\
					-nSamplesPerFragment $nSamplesPerFragment\
					-threshold $threshold\
					-MODEL1_DIR $MODEL1_DIR\
					-MODEL2_DIR $MODEL2_DIR\
					-mode $MODE&
				((j++))
		done 
		wait
		rm -r ${fragName}*
	done
	wait
done
# remove related data for should't conflict next data version
rm -r $fragBase*
echo All done

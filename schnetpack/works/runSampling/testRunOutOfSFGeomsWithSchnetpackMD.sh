#! /usr/bin/bash

NSAMPLE=1000
MDTYPE="classic"
MODE="active"
# MODE="ensembleActive"
mof_num=10
len_data=58754
nSamplesPerFragment=50
threshold=0.4

BASEDIR="/truba_scratch/yzorlu/deepMOF/HDNNP/"
PYTHON38DIR="/truba/home/yzorlu/miniconda3/bin/"
# PYTHONDIR="/truba/home/yzorlu/miniconda3/bin/"
SCRIPTDIR="/truba_scratch/yzorlu/deepMOF/HDNNP/schnetpack"
# MODEL1_DIR=$BASEDIR/schnetpack/runTraining/hdnnBehler_l3n100_rho001_r20a5_lr0001_bs1_IRMOFseries${mof_num}_merged_${len_data}_ev
# MODEL2_DIR=$BASEDIR/schnetpack/runTraining/hdnnBehler_l3n50_rho001_r20a5_lr0001_bs1_IRMOFseries${mof_num}_merged_${len_data}_ev
MODEL1_DIR=$BASEDIR/schnetpack/runTraining/schnet_l3_basis96_filter64_interact3_gaussian20_rho001_lr00001_bs1_cutoff_60_withoutStress_aseEnv_IRMOFseries${mof_num}_merged_${len_data}_ev
MODEL2_DIR=$BASEDIR/schnetpack/runTraining/schnet_l4_basis96_filter64_interact3_gaussian20_rho001_lr00001_bs1_cutoff_60_withoutStress_aseEnv_IRMOFseries${mof_num}_merged_${len_data}_ev

dbBase="nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling_IRMOFseries${mof_num}_merged_50000_ev"

outOfGeomsDIR=$BASEDIR/prepare_data/test_outOfSFGeoms
GPUIDS=(0 1)

for k in ${mof_num}
do
	fragBase="irmofseries${k}_f"
for i in 6
do
	fragName=${fragBase}${i}
	echo $fragName
	# MOLPATH=$BASEDIR/prepare_data/geomFiles/IRMOFSeries/fragments/$fragName.xyz
	$PYTHON38DIR/python -u $SCRIPTDIR/calculateSF.py -fragName $fragName\
	        -dbBase $dbBase
	
	maxMinSFPath=$PWD/$fragName"_MinMaxValues.csv"
	
	# counter=1
	j=0
	for TEMP in {300..300..300} 
	do
			CUDA_VISIBLE_DEVICES=${GPUIDS[$(((j)%${#GPUIDS[@]}))]} # to assing job each GPU
			counter=0
			# while [ $counter -le 0 ]
			# do
				$PYTHON38DIR/python -u $SCRIPTDIR/run_md_withSchnetpack_active.py \
					-mdtype $MDTYPE\
					-n $NSAMPLE -temp $TEMP -fragName $fragName\
					-maxMinSFPath $maxMinSFPath\
					-outOfGeomsDIR $outOfGeomsDIR\
					-mode $MODE\
					-nSamplesPerFragment $nSamplesPerFragment\
					-threshold $threshold\
					-MODEL1_DIR $MODEL1_DIR\
					-MODEL2_DIR $MODEL2_DIR
				((counter++))
				# rm -r "${fragName}_md_${TEMP}K${MDTYPE}"
				# rm -r ${fragName}*
			# done
			((j++))
	done 
done
done
echo All done

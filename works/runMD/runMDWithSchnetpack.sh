#! /usr/bin/bash
CUDA_VISIBLE_DEVICES="0"

NSAMPLE=1000000
MDTYPE="md"
MODE="md"
# MODE="ensamble"
mof_num=10
descrip_word="_NPT_111cell_allNNP_without5"
# len_data=97355

BASEDIR="/truba_scratch/yzorlu/deepMOF/HDNNP/prepare_data"
PYTHON38DIR="/truba/home/yzorlu/miniconda3/bin/"
SCRIPTDIR="/truba_scratch/yzorlu/deepMOF/HDNNP/schnetpack"


molName=obabel_IRMOF${mof_num}
echo $molName
MOLPATH=$BASEDIR/geomFiles/IRMOFSeries/cif_files/$molName.cif
# MODEL1_DIR="hdnnBehler_l3n50_rho001_r20a5_lr0001_bs1_IRMOFseries${mof_num}_merged_${len_data}_ev"
MODEL1_DIR="schnet_l3_basis96_filter64_interact3_gaussian20_rho001_lr00001_bs1_cutoff_60_withoutStress_aseEnv_test_IRMOFseries1_4_6_7_10_merged_173014_ev"
# MODEL2_DIR="hdnnBehler_l3n50_rho001_r20a5_lr0001_bs1_IRMOFseries${mof_num}_merged_${len_data}_ev"

$PYTHON38DIR/python -u $SCRIPTDIR/run_md_withSchnetpack.py \
	-mol $MOLPATH -mdtype $MDTYPE\
	-n $NSAMPLE -temp 100\
	-init_temp 100\
	-calc_mode $MODE\
	-descrip_word $descrip_word\
	-opt yes\
        -restart no\
	-MODEL1_DIR $MODEL1_DIR
	# -MODEL2_DIR $MODEL2_DIR

echo All done

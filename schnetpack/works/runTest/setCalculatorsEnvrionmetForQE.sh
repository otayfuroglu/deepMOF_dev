#! /usr/bin/bash

#unload current modules
module unload centos7.3/lib/openmpi/3.0.0-gcc-7.0.1

# set environmet variable for Esperesso
source /truba/sw/centos7.3/comp/intel/PS2013-SP1/bin/compilervars.sh intel64
module load centos7.3/app/espresso/6.1-openmpi-1.8.8-mkl-PS2013-GOLD
module load centos7.3/lib/openmpi/1.8.8-intel-PS2013
export ESPRESSO_DIR=/truba/sw/centos7.3/app/espresso/6.1-openmpi-1.8.8-mkl-PS2013-E5V4
export ESPRESSO_PSEUDO=/truba/home/otayfuroglu/pseudo_pot/qe/all_lda_UPF_v1.5/


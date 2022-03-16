#! /usr/bin/bash

#unload current modules
module unload centos7.3/lib/openmpi/3.0.0-gcc-7.0.1

#set environmet variable for siesta
module load centos7.3/lib/openmpi/1.8.8-gcc-4.8.5
export SIESTA_PATH="/truba/sw/centos7.3/app/siesta/4.1-b3/bin/siesta-openmpi-1.8.8-gcc-4.8.5-E5V4"
export SIESTA_PP_PATH="/truba/home/otayfuroglu/pseudo_pot/siesta/cornell/GGA"



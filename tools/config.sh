#!/bin/bash

# cpu
export SZ_Huffman=~/project/sz3/install/bin/sz3
export SZ3_CONFIG=~/project/sz3/tools/sz3/sz3.config
export FSE=~/project/FSE-THR/fse
export ADT_PATH=~/project/IntZip/sample_data
export ADM16_cpu=~/project/mans/build/bin/cpu/adm_compress
export ADM32_cpu=~/project/mans/build/bin/cpu/adm_compress
export PANS_CMP_cpu=~/project/mans/build/bin/cpu/cpuans_compress
export PANS_DECMP_cpu=~/project/mans/build/bin/cpu/cpuans_decompress

# nv
export NVCOMP_PATH=~/project/nvcomp/bin
export ADM16_nv=~/project/mans/build/bin/nv/nvadm_compress
export ADM32_nv=~/project/mans/build/bin/nv/nvadm_compress
export PANS_CMP_nv=~/project/mans/build/bin/nv/cudaans_compress
export PANS_DECMP_nv=~/project/mans/build/bin/nv/cudaans_decompress


# output file
export TEST_DIR_u2=/data/hwj/testdata/u2
export TEST_DIR_u4=/data/hwj/testdata/u4
export OUTPUT_DIR=./output

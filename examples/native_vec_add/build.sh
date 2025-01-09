#!/bin/bash
#
ml load cuda/12.2
ml load gcc/11.2.1
ml load cmake/3.23
ml load python/3.8
#export PATH="/usr/workspace/LExperts/DaCe/usr_18.1.8/blueos_3_ppc64le_ib_p9/bin/:$PATH"
ml load clang/18.1.8-cuda-11.8.0-gcc-11.2.1

cmake .. \
-DRR_Dir=$(realpath ../../../build_lassen/install/lib/cmake) \
-DCMAKE_BUILD_TYPE=Relwithdebinfo \
-DCMAKE_INSTALL_PREFIX=$installDir \
-DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_C_COMPILER=clang \
-DCMAKE_CUDA_COMPILER=clang++ \
-DCMAKE_C_COMPILER=clang \
-DCMAKE_CUDA_COMPILER=clang++ \
-DLLVM_INSTALL_DIR=$(dirname $(dirname $(which clang))) \
-DENABLE_CUDA=On \
-DCMAKE_EXPORT_COMPILE_COMMANDS=on ../


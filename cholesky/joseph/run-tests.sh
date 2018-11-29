#!/bin/bash

tmp_result_folder="results_`date '+%Y%m%d_%H%M%S'`"
m_size=5120
b_size=512
b_check=1
n_ranks=2
n_threads=4

#for target in intel clang chameleon
for target in intel
do
  # load default modules
  module purge
  module load DEVELOP

  # load target specific compiler and libraries
  while IFS='' read -r line || [[ -n "$line" ]]; do
    if  [[ $line == LOAD_COMPILER* ]] || [[ $line == LOAD_LIBS* ]] ; then
      eval "$line"
    fi
  done < "flags_${target}.def"
  module load ${LOAD_COMPILER}
  module load intelmpi/2018.3
  module load ${LOAD_LIBS}
  module li

  # make result folder once
  mkdir -p ${tmp_result_folder}
  
  # run single comm tests
  for version in ch_${target}_single ch_${target}_single_noyield
  do
        OMP_NUM_THREADS=${n_threads} \
        NUM_RANKS=${n_ranks} \
        PROG_EXE=${version} \
        TARGET=${target} \
        MATRIX_SIZE=${m_size} \
        BLOCK_SIZE=${b_size} \
        BOOL_CHECK=${b_check} \
        make -C singlecom-deps run 2>&1 | tee "${tmp_result_folder}/${version}.txt"
  done

  # run fine grained tests
  for version in ch_${target}_fine
  do
        OMP_NUM_THREADS=${n_threads} \
        NUM_RANKS=${n_ranks} \
        PROG_EXE=${version} \
        TARGET=${target} \
        MATRIX_SIZE=${m_size} \
        BLOCK_SIZE=${b_size} \
        BOOL_CHECK=${b_check} \
        make -C fine-deps run 2>&1 | tee "${tmp_result_folder}/${version}.txt"
  done
  
  # Note: for now didnt perform perrank (reason: see IWOMP 18 paper from Joseph)

done

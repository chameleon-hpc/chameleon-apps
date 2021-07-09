#!/usr/local_rwth/bin/zsh

if [ "${RUN_LIKWID}" = "1" ]; then
    module load likwid
    #-o ${TMP_NAME_RUN}_hwc_R${PMI_RANK}.csv -O
    LIKW_EXT="likwid-perfctr -f -m -c N:0-$((OMP_NUM_THREADS-1)) -T 500us -g L2CACHE -g L3CACHE -g L2 -g L3"
fi

# remember current cpuset for process
CUR_CPUSET=$(cut -d':' -f2 <<< $(taskset -c -p $(echo $$)) | xargs)
# echo "${PMI_RANK}: CUR_CPUSET = ${CUR_CPUSET}"

if [ "${RUN_LIKWID}" = "1" ]; then
    echo "Command executed for rank ${PMI_RANK}: ${LIKW_EXT} taskset -c ${CUR_CPUSET} ${DIR_MXM_EXAMPLE}/${EXE_NAME} ${GRANU} ${MXM_PARAMS}"
    ${LIKW_EXT} taskset -c ${CUR_CPUSET} ${DIR_MXM_EXAMPLE}/${EXE_NAME} ${GRANU} ${MXM_PARAMS}
else
    echo "Command executed for rank ${PMI_RANK}: ${LIKW_EXT} ${DIR_MXM_EXAMPLE}/${EXE_NAME} ${GRANU} ${MXM_PARAMS}"
    ${LIKW_EXT} ${DIR_MXM_EXAMPLE}/${EXE_NAME} ${GRANU} ${MXM_PARAMS}
fi

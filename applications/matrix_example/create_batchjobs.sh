#!/usr/local_rwth/bin/zsh
#SBATCH --time=00:02:00

# ========================================
# Global Settings
# ========================================
export CUR_DATE_STR="$(date +"%Y%m%d_%H%M%S")"
export RUN_LIKWID=1
IMBALANCED=(0 1)

#for imb in "${IMBALANCED[@]}"
#do
    #export IS_IMBALANCED=${imb}

    # 2 rank job - shared memory
    #export N_PROCS=2
    #sbatch --nodes=1 --ntasks-per-node=2 --cpus-per-task=24 --job-name=mxm_2procs_sm --output=mxm_2procs_sm.%J.txt --export=IS_IMBALANCED,RUN_LIKWID,CUR_DATE_STR,N_PROCS run_experiments.sh

    # 4 rank job - shared memory
    export N_PROCS=4
    sbatch --nodes=1 --ntasks-per-node=4 --cpus-per-task=12 --job-name=mxm_4procs_sm --output=mxm_4procs_sm.%J.txt --export=RUN_LIKWID,CUR_DATE_STR,N_PROCS run_experiments.sh

    # 4 rank job - distributed memory
    #export N_PROCS=4
    #sbatch --nodes=2 --ntasks-per-node=2 --cpus-per-task=24 --job-name=mxm_4procs_dm --output=mxm_4procs_dm.%J.txt --export=IS_IMBALANCED,RUN_LIKWID,CUR_DATE_STR,N_PROCS run_experiments.sh
#done
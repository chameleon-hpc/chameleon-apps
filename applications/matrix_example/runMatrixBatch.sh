#!/usr/local_rwth/bin/zsh

### Job name
#SBATCH --job-name=matrix_example-batch

#SBATCH --output=output_%J.txt

###SBATCH --account=lect0034
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --partition=c18m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12

###SBATCH --mail-user=julian.schacht@rwth-aachen.de

source ~/.zshrc

CHAMELEON_TOOL_LIBRARIES=/home/ey186093/GitLab/jusch_chameleon-apps/tools/tool_task_balancing/tool.so;
RUN_SETTINGS_SLURM='OMP_PLACES=cores OMP_PROC_BIND=close I_MPI_FABRICS="shm"';
MPI_EXPORT_VARS_SLURM='--export=PATH,CPLUS_INCLUDE_PATH,C_INCLUDE_PATH,CPATH,INCLUDE,LD_LIBRARY_PATH,LIBRARY_PATH,I_MPI_DEBUG,I_MPI_TMI_NBITS_RANK,OMP_NUM_THREADS,OMP_PLACES,OMP_PROC_BIND,KMP_AFFINITY,I_MPI_FABRICS,MIN_ABS_LOAD_IMBALANCE_BEFORE_MIGRATION,MIN_LOCAL_TASKS_IN_QUEUE_BEFORE_MIGRATION,MIN_REL_LOAD_IMBALANCE_BEFORE_MIGRATION,MAX_TASKS_PER_RANK_TO_MIGRATE_AT_ONCE,PERCENTAGE_DIFF_TASKS_TO_MIGRATE,ENABLE_TRACE_FROM_SYNC_CYCLE,ENABLE_TRACE_TO_SYNC_CYCLE';
PROG= 'main';
NUM_TASKS=1200;

mkdir -p "/home/ey186093/output/$(date '+%Y-%m-%d')";

for MATRIX_SIZE in {100..200..100}; do
  echo "AUSGABE: ${RUN_SETTINGS_SLURM} ${MPIEXEC} ${FLAGS_MPI_BATCH} ${MPI_EXPORT_VARS_SLURM} ${PROG} ${MATRIX_SIZE} ${NUM_TASKS} ${NUM_TASKS} ${NUM_TASKS} ${NUM_TASKS}";
  $RUN_SETTINGS_SLURM $MPIEXEC $FLAGS_MPI_BATCH $MPI_EXPORT_VARS_SLURM $PROG $MATRIX_SIZE $NUM_TASKS $NUM_TASKS $NUM_TASKS $NUM_TASKS
  ###if grep -q "FAILED" "log.log"; then
    ###echo "ERROR: One test failed";
    ###break;
  ###fi

  rename ".txt" "_S${MATRIX_SIZE}.txt" /home/ey186093/output/*;
  mv /home/ey186093/output/*.txt "/home/ey186093/output/$(date '+%Y-%m-%d')";
done
#!/usr/local_rwth/bin/zsh
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --partition=c18m
#SBATCH --hwctr=likwid

# =============== Load desired modules
source ~/.zshrc
module load chameleon

# =============== Settings & environment variables
N_PROCS=${N_PROCS:-2}
#IS_IMBALANCED=${IS_IMBALANCED:-0}
RUN_LIKWID=${RUN_LIKWID:-1}
N_REPETITIONS=${N_REPETITIONS:-1}
CUR_DATE_STR=${CUR_DATE_STR:-"$(date +"%Y%m%d_%H%M%S")"}
EXE_NAME=${EXE_NAME:-"matrix_app.exe"}

# ============== Project variables
CHAMELEON_TOOL_LIBRARIES=/home/ey186093/GitLab/jusch_chameleon-apps/tools/tool_task_balancing/tool_UNI.so;
LIKWID_GROUPS=("CLOCK" "CYCLE_ACTIVITY" "CYCLE_STALLS" "DATA" "FLOPS_DP" "FLOPS_AVX" "L2" "L2CACHE" "L3" "L3CACHE" "MEM_DP" "TLB_DATA");
DIR_MXM_EXAMPLE=${DIR_MXM_EXAMPLE:-/rwthfs/rz/cluster/home/ey186093/GitLab/jusch_chameleon-apps/applications/matrix_example}
export OMP_NUM_THREADS=4;
export OMP_PLACES=cores;
export OMP_PROC_BIND=close;
export I_MPI_PIN=1;
export I_MPI_PIN_DOMAIN=auto;

mkdir -p "/home/ey186093/output/$(date '+%Y-%m-%d')";
mkdir -p "/home/ey186093/output/$(date '+%Y-%m-%d')/rank";
mkdir -p "/home/ey186093/output/$(date '+%Y-%m-%d')/runtime";
RESULT_DIR="/home/ey186093/output/$(date '+%Y-%m-%d')";

echo "Script Dir: ${CUR_DIR}"
echo "Result Dir: ${RESULT_DIR}"
echo "Repetitions: ${N_REPETITIONS}"

#=== nr of threads, matrix sizes, workloads, sequences
N_THREADS=(8)
MSS=(200 300 350 400 450)

##DIR_RESULT="/rwthfs/rz/cluster/home/ey186093/output/${CUR_DATE_STR}_results/imb${IS_IMBALANCED}_${N_PROCS}procs_${SLURM_NTASKS_PER_NODE}pernode"
##mkdir -p ${DIR_RESULT}

# export necessary stuff that is used in wrapper script
export EXE_NAME
export DIR_MXM_EXAMPLE
export RUN_LIKWID
export CHAMELEON_TOOL_LIBRARIES

export MIN_REL_LOAD_IMBALANCE_BEFORE_MIGRATION=0.1
export MAX_TASKS_PER_RANK_TO_MIGRATE_AT_ONCE=1
export PERCENTAGE_DIFF_TASKS_TO_MIGRATE=0.3
export MPI_EXEC_CMD="${RUN_SETTINGS_SLURM} ${MPIEXEC} ${FLAGS_MPI_BATCH}"

for nt in "${N_THREADS[@]}"
do
  export OMP_NUM_THREADS=${nt}
  export MIN_LOCAL_TASKS_IN_QUEUE_BEFORE_MIGRATION=${nt}


  for matrix_size in "${MSS[@]}";
  do

    for workload in {10..10..10};
    do

      for SEQUENCE in {1..1..1};
      do

        case ${SEQUENCE} in
          1)
            a=$((1* $workload));
            b=$((2* $workload));
            c=$((4* $workload));
            d=$((8* $workload));
          ;;

          2)
            a=0;
            b=$((2* $workload));
            c=$((4* $workload));
            d=$((8* $workload));
          ;;

          3)
            a=$((1* $workload));
            b=0;
            c=$((4* $workload));
            d=$((8* $workload));
          ;;

          4)
            a=$((1* $workload));
            b=$((2* $workload));
            c=0;
            d=$((8* $workload));
          ;;

          5)
            a=$((1* $workload));
            b=$((2* $workload));
            c=$((4* $workload));
            d=0;
          ;;

          6)
            a=0;
            b=0;
            c=$((4* $workload));
            d=$((8* $workload));
          ;;

          7)
            a=$((1* $workload));
            b=0;
            c=0;
            d=$((8* $workload));
          ;;

          8)
            a=$((1* $workload));
            b=$((2* $workload));
            c=0;
            d=0;
          ;;

          9)
            a=0;
            b=$((2* $workload));
            c=$((4* $workload));
            d=0;
          ;;

          10)
            a=0;
            b=0;
            c=0;
            d=$((8* $workload));
          ;;

          11)
            a=$((8* $workload));
            b=$a;
            c=$a;
            d=$a;
          ;;

          12)
            a=$((8* $workload));
            b=0;
            c=$a;
            d=$a;
          ;;

          13)
            a=$((8* $workload));
            b=0;
            c=0;
            d=$a;
            ;;

          *)
            echo "Unsupported case of workloads"
            exit
          ;;
        esac

        MXM_PARAMS="$a $b $c $d";
        export MXM_PARAMS
        export GRANU=${matrix_size}

        echo "Running experiments for matrix_size ${matrix_size}, workload ${workload}, sequence ${SEQUENCE} and n_threads=${nt}"
        for r in {1..${N_REPETITIONS}}
        do
          echo "Starting repetition: ${r}"
          export TMP_NAME_RUN="${RESULT_DIR}/results_${matrix_size}_${workload}_${SEQUENCE}_${nt}t_${r}"
          eval "${MPI_EXEC_CMD} ${MPI_EXPORT_VARS_SLURM} ./wrapper.sh" &> ${TMP_NAME_RUN}.log

          touch "${RESULT_DIR}/UNI_S_${matrix_size}_W_${a}_${b}_${c}_${d}.csv";
          touch "${RESULT_DIR}/UNI_S_${matrix_size}_W_${a}_${b}_${c}_${d}_R.csv";

          cat "/home/ey186093/output/.head/HEAD.csv" >> "${RESULT_DIR}/UNI_S_${matrix_size}_W_${a}_${b}_${c}_${d}.csv";
          cat "/home/ey186093/output/.head/HEAD_R.csv" >> "${RESULT_DIR}/UNI_S_${matrix_size}_W_${a}_${b}_${c}_${d}_R.csv";

          rm "/home/ey186093/output/.head/HEAD.csv";
          rm "/home/ey186093/output/.head/HEAD_R.csv";

          for file in /home/ey186093/output/.runtime/*.csv; do
            cat "${file}" >> "${RESULT_DIR}/UNI_S_${matrix_size}_W_${a}_${b}_${c}_${d}.csv";
            rm "${file}";
          done

          for file in /home/ey186093/output/.rank/*.csv; do
            cat "${file}" >> "${RESULT_DIR}/UNI_S_${matrix_size}_W_${a}_${b}_${c}_${d}_R.csv";
            rm "${file}";
          done

          mv "${RESULT_DIR}/UNI_S_${matrix_size}_W_${a}_${b}_${c}_${d}.csv" "/home/ey186093/output/$(date '+%Y-%m-%d')/runtime";
          mv "${RESULT_DIR}/UNI_S_${matrix_size}_W_${a}_${b}_${c}_${d}_R.csv" "/home/ey186093/output/$(date '+%Y-%m-%d')/rank";

        done

      done
    done
  done
done

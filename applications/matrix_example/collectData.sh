#!/usr/local_rwth/bin/zsh

#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --account=thes0970
#SBATCH --job-name=THESIS_uniform_matrix
#SBATCH --output=THESIS_uniform_matrix_output.%J.txt

module use -a /home/ey186093/.modules
module load chameleon

export CHAMELEON_TOOL_LIBRARIES=/home/ey186093/GitLab/jusch_chameleon-apps/tools/tool_task_balancing/tool.so;
export OMP_NUM_THREADS=4;
export OMP_PLACES=cores;
export OMP_PROC_BIND=close;
export I_MPI_PIN=1;
export I_MPI_PIN_DOMAIN=auto;

mkdir -p "/home/ey186093/output/$(date '+%Y-%m-%d')";
mkdir -p "/home/ey186093/output/$(date '+%Y-%m-%d')/node";
mkdir -p "/home/ey186093/output/$(date '+%Y-%m-%d')/runtime";

### 100.600.100
for matrix_size in {100..600..100}; do
  ### 10..50..10
  for workload in {10..50..10}; do
    for SEQUENCE in {1..13..1}; do

      if [ $SEQUENCE -eq 1 ]
        then
          a=$((1* $workload));
          b=$((2* $workload));
          c=$((4* $workload));
          d=$((8* $workload));
      elif [ $SEQUENCE -eq 2 ]
        then
          a=0;
          b=$((2* $workload));
          c=$((4* $workload));
          d=$((8* $workload));
      elif [ $SEQUENCE -eq 3 ]
        then
          a=$((1* $workload));
          b=0;
          c=$((4* $workload));
          d=$((8* $workload));
      elif [ $SEQUENCE -eq 4 ]
        then
          a=$((1* $workload));
          b=$((2* $workload));
          c=0;
          d=$((8* $workload));
      elif [ $SEQUENCE -eq 5 ]
        then
          a=$((1* $workload));
          b=$((2* $workload));
          c=$((4* $workload));
          d=0;
      elif [ $SEQUENCE -eq 6 ]
        then
          a=0;
          b=0;
          c=$((4* $workload));
          d=$((8* $workload));
      elif [ $SEQUENCE -eq 7 ]
        then
          a=$((1* $workload));
          b=0;
          c=0;
          d=$((8* $workload));
      elif [ $SEQUENCE -eq 8 ]
        then
          a=$((1* $workload));
          b=$((2* $workload));
          c=0;
          d=0;
      elif [ $SEQUENCE -eq 9 ]
        then
          a=0;
          b=$((2* $workload));
          c=$((4* $workload));
          d=0;
      elif [ $SEQUENCE -eq 10 ]
        then
          a=0;
          b=0;
          c=0;
          d=$((8* $workload));
      elif [ $SEQUENCE -eq 11 ]
        then
            a=$((8* $workload));
            b=$a;
            c=$a;
            d=$a;
      elif [ $SEQUENCE -eq 12 ]
        then
          a=$((8* $workload));
          b=0;
          c=$a;
          d=$a;
      elif [ $SEQUENCE -eq 13 ]
        then
          a=$((8* $workload));
          b=0;
          c=0;
          d=$a;
      fi

      mpiexec.hydra -np 4 -genvall main $matrix_size $a $b $c $d 2>&1 | tee log.log;

      if grep -q "FAILED" "log.log"; then
        echo "ERROR: One test failed";
        break;
      fi

      touch "RESULT_S_${matrix_size}_W_${a}_${b}_${c}_${d}.csv";
      touch "RESULT_S_${matrix_size}_W_${a}_${b}_${c}_${d}_N.csv";

      cat "/home/ey186093/output/.head/HEAD.csv" >> "RESULT_S_${matrix_size}_W_${a}_${b}_${c}_${d}.csv";
      cat "/home/ey186093/output/.head/HEAD_N.csv" >> "RESULT_S_${matrix_size}_W_${a}_${b}_${c}_${d}_N.csv";
      rm "/home/ey186093/output/.head/HEAD.csv";
      rm "/home/ey186093/output/.head/HEAD_N.csv";

      ####echo 'RUNTIME_ms;RMIN;RMAX;RMEAN;#ARGS;#IARGS;SIARGS;SIMIN;SIMAX;SIMEAN;SIOVER;#OARGS;SOARGS;SOMIN;SOMAX;SOMEAN;SOVER;OVSITA;#NOTA;ROVER;' >> "RESULT_S${matrix_size}W_${a}_${b}_${c}_${d}.csv";

      for file in /home/ey186093/output/.runtime/*.csv; do
        ###if [[ "${file}" == *"R0"* ]]; then
          ###echo "----R0---" >> "RESULT_S${matrix_size}W_${a}_${b}_${c}_${d}.csv";
        ####elif [[ "${file}" == *"R1"* ]]; then
          ###echo "---R1---" >> "RESULT_S${matrix_size}W_${a}_${b}_${c}_${d}.csv";
        ###elif [[ "{$file}" == *"R2"* ]]; then
          ###echo "---R2---" >> "RESULT_S${matrix_size}W_${a}_${b}_${c}_${d}.csv";
        ###elif [[ "{$file}" == *"R3"* ]]; then
          ###echo "---R3---" >> "RESULT_S${matrix_size}W_${a}_${b}_${c}_${d}.csv";
        ###fi

        cat "${file}" >> "RESULT_S_${matrix_size}_W_${a}_${b}_${c}_${d}.csv";
        rm "${file}";
      done

      for file in /home/ey186093/output/.node/*.csv; do
        ###if [[ "${file}" == *"R0"* ]]; then
          ###echo "----R0---" >> "RESULT_S${matrix_size}W_${a}_${b}_${c}_${d}.csv";
        ####elif [[ "${file}" == *"R1"* ]]; then
          ###echo "---R1---" >> "RESULT_S${matrix_size}W_${a}_${b}_${c}_${d}.csv";
        ###elif [[ "{$file}" == *"R2"* ]]; then
          ###echo "---R2---" >> "RESULT_S${matrix_size}W_${a}_${b}_${c}_${d}.csv";
        ###elif [[ "{$file}" == *"R3"* ]]; then
          ###echo "---R3---" >> "RESULT_S${matrix_size}W_${a}_${b}_${c}_${d}.csv";
        ###fi

        cat "${file}" >> "RESULT_S_${matrix_size}_W_${a}_${b}_${c}_${d}_N.csv";
        rm "${file}";
      done

      mv "RESULT_S_${matrix_size}_W_${a}_${b}_${c}_${d}.csv" "/home/ey186093/output/$(date '+%Y-%m-%d')/runtime";
      mv "RESULT_S_${matrix_size}_W_${a}_${b}_${c}_${d}_N.csv" "/home/ey186093/output/$(date '+%Y-%m-%d')/node";

    done
  done
done
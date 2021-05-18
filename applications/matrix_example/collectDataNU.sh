#!/usr/local_rwth/bin/zsh

#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --account=thes0970
#SBATCH --job-name=THESIS_nonuniform_matrix
#SBATCH --output=THESIS_nonuniform_matrix_output.%J.txt

module use -a /home/ey186093/.modules
module load chameleon

export CHAMELEON_TOOL_LIBRARIES=/home/ey186093/GitLab/jusch_chameleon-apps/tools/tool_task_balancing/tool.so;
export OMP_NUM_THREADS=4;
export OMP_PLACES=cores;
export OMP_PROC_BIND=close;
export I_MPI_PIN=1;
export I_MPI_PIN_DOMAIN=auto;

mkdir -p "/home/ey186093/output/$(date '+%Y-%m-%d')_NU";
mkdir -p "/home/ey186093/output/$(date '+%Y-%m-%d')_NU/node";
mkdir -p "/home/ey186093/output/$(date '+%Y-%m-%d')_NU/runtime";


### Number of matrices
export NOM=50;

for iterator in {0..255..1}; do
  mS=(50 100 150 200 450 500 550 600);
  nOM=(0 0 0 0 0 0 0 0);

  for i in {1..8..1}; do
    nOM[$i]=$NOM;
  done

  for CASE in {1..3..1}; do

    if [ $CASE -eq 1 ]
          then
            sort=(0 0 0 1);
    elif [ $CASE -eq 2 ]
          then
            sort=(0 0 1 1);
    elif [ $CASE -eq 3 ]
          then
            sort=(0 1 1 1);
    fi

    switch=$(printf "%.8d\n" $(echo "obase=2;$iterator" | bc));
    switchArray=("${(@s/:/)switch}");

    ######read -a switchArray <<< $(echo $switch| sed 's/./& /g');

    for j in {1..8..1}; do
      #### $((7-$j))
      if [[ ${switchArray[$((8-$j))]} == 1 ]]; then
        nOM[$j]=0;
      fi
    done

    mpiexec.hydra -np 4 ./main non-uniform ${mS[1]},${mS[2]},${mS[3]},${mS[4]},${mS[5]},${mS[6]},${mS[7]},${mS[8]} ${nOM[1]},${nOM[2]},${nOM[3]},${nOM[4]},${nOM[5]},${nOM[6]},${nOM[7]},${nOM[8]} "${sort[1]}" "${sort[2]}" "${sort[3]}" "${sort[4]}" 2>&1 | tee log.log;

    if grep -q -i "FAILED" "log.log"; then
      echo "ERROR: One test failed - S_${sort[0]}_${sort[1]}_${sort[2]}_${sort[3]}_MS_${mS[0]}_${mS[1]}_${mS[2]}_${mS[3]}_${mS[4]}_${mS[5]}_${mS[6]}_${mS[7]}";
      break 4;
    fi
    if grep -q -i "BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES" "log.log"; then
      echo "ERROR: One test failed - S_${sort[0]}_${sort[1]}_${sort[2]}_${sort[3]}_MS_${mS[0]}_${mS[1]}_${mS[2]}_${mS[3]}_${mS[4]}_${mS[5]}_${mS[6]}_${mS[7]}";
      break 4;
    fi

    ##touch log.txt;
    ##echo "${mS[@]}" >> log.txt;
    ##echo $nOM >> log.txt;
    ##echo "${sort[@]}" >> log.txt;
    touch "RESULT_S_${sort[1]}_${sort[2]}_${sort[3]}_${sort[4]}_MS_${NOM}_I_${iterator}.csv";
    touch "RESULT_S_${sort[1]}_${sort[2]}_${sort[3]}_${sort[4]}_MS_${NOM}_I_${iterator}_N.csv";

    cat "/home/ey186093/output/.head/HEAD.csv" >> "RESULT_S_${sort[1]}_${sort[2]}_${sort[3]}_${sort[4]}_MS_${NOM}_I_${iterator}.csv";
    cat "/home/ey186093/output/.head/HEAD_N.csv" >> "RESULT_S_${sort[1]}_${sort[2]}_${sort[3]}_${sort[4]}_MS_${NOM}_I_${iterator}_N.csv";

    rm "/home/ey186093/output/.head/HEAD.csv";
    rm "/home/ey186093/output/.head/HEAD_N.csv";

    for file in /home/ey186093/output/.runtime/*.csv; do
      ###if [[ "${file}" == *"R0"* ]]; then
        ###echo "---R0---" >> "RESULT_S_${sort[1]}_${sort[2]}_${sort[3]}_${sort[4]}_MS_${NOM}_I_${iterator}.csv";
      ###elif [[ "${file}" == *"R1"* ]]; then
        ###echo "---R1---" >> "RESULT_S_${sort[1]}_${sort[2]}_${sort[3]}_${sort[4]}_MS_${NOM}_I_${iterator}.csv";
      ###elif [[ "${file}" == *"R2"* ]]; then
        ###echo "---R2---" >> "RESULT_S_${sort[1]}_${sort[2]}_${sort[3]}_${sort[4]}_MS_${NOM}_I_${iterator}.csv";
      ###elif [[ "${file}" == *"R3"* ]]; then
        ###echo "---R3---" >> "RESULT_S_${sort[1]}_${sort[2]}_${sort[3]}_${sort[4]}_MS_${NOM}_I_${iterator}.csv";
      ###fi
      cat "${file}" >> "RESULT_S_${sort[1]}_${sort[2]}_${sort[3]}_${sort[4]}_MS_${NOM}_I_${iterator}.csv";
      rm "${file}";
    done

    for file in /home/ey186093/output/.node/*.csv; do
      ###if [[ "${file}" == *"R0"* ]]; then
        ###echo "---R0---" >> "RESULT_S_${sort[1]}_${sort[2]}_${sort[3]}_${sort[4]}_MS_${NOM}_I_${iterator}.csv";
      ###elif [[ "${file}" == *"R1"* ]]; then
        ###echo "---R1---" >> "RESULT_S_${sort[1]}_${sort[2]}_${sort[3]}_${sort[4]}_MS_${NOM}_I_${iterator}.csv";
      ###elif [[ "${file}" == *"R2"* ]]; then
        ###echo "---R2---" >> "RESULT_S_${sort[1]}_${sort[2]}_${sort[3]}_${sort[4]}_MS_${NOM}_I_${iterator}.csv";
      ###elif [[ "${file}" == *"R3"* ]]; then
        ###echo "---R3---" >> "RESULT_S_${sort[1]}_${sort[2]}_${sort[3]}_${sort[4]}_MS_${NOM}_I_${iterator}.csv";
      ###fi
      cat "${file}" >> "RESULT_S_${sort[1]}_${sort[2]}_${sort[3]}_${sort[4]}_MS_${NOM}_I_${iterator}_N.csv";
      rm "${file}";
    done

    mv "RESULT_S_${sort[1]}_${sort[2]}_${sort[3]}_${sort[4]}_MS_${NOM}_I_${iterator}_N.csv" "/home/ey186093/output/$(date '+%Y-%m-%d')_NU/node";
    mv "RESULT_S_${sort[1]}_${sort[2]}_${sort[3]}_${sort[4]}_MS_${NOM}_I_${iterator}.csv" "/home/ey186093/output/$(date '+%Y-%m-%d')_NU/runtime";
  done
done



#!/usr/local_rwth/bin/zsh
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --partition=c18m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --hwctr=likwid

# get current script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# load required modules
source ~/.zshrc
module load chameleon
module load likwid
module load DEV-TOOLS papi
module list

# settings
export OMP_NUM_THREADS=11
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export MATRIX_EXE="/rwthfs/rz/cluster/work/jk869269/repos/hpc-projects/chameleon/chameleon-apps/applications/matrix_example/matrix_app.exe"
export NUM_TASKS=500
export LIKWID_FORCE=1
export LIKWID_DEBUG=0
export NUM_REP=10
export OUTPUT_RESULTS=0

export RESULT_DIR="${SCRIPT_DIR}/results"
mkdir -p ${RESULT_DIR}

#export LIKWID_EVENTS="INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,CPU_CLK_UNHALTED_REF:FIXC2,L1D_REPLACEMENT:PMC0,L1D_M_EVICT:PMC1,L2_LINES_IN_ALL:PMC2,L2_TRANS_L2_WB:PMC3,FP_ARITH_INST_RETIRED_128B_PACKED_DOUBLE:PMC4,FP_ARITH_INST_RETIRED_SCALAR_DOUBLE:PMC5,FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE:PMC6,FP_ARITH_INST_RETIRED_512B_PACKED_DOUBLE:PMC7,TEMP_CORE:TMP0"
export LIKWID_EVENTS="INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,CPU_CLK_UNHALTED_REF:FIXC2,L2_LINES_IN_ALL:PMC2,L2_TRANS_L2_WB:PMC3,FP_ARITH_INST_RETIRED_128B_PACKED_DOUBLE:PMC4,FP_ARITH_INST_RETIRED_SCALAR_DOUBLE:PMC5,FP_ARITH_INST_RETIRED_256B_PACKED_DOUBLE:PMC6,FP_ARITH_INST_RETIRED_512B_PACKED_DOUBLE:PMC7"

M_SIZES=(10 20 30 40 50 60 70 80 90 100 110 120 150 200 250 300 350 400 450 500 550 600 650 700)
VERSIONS=("no-tool" "likwid" "papi" "perf")

run_experiment() {
    export RESULT_PATH="${RESULT_DIR}/hwc_${LOG_POSTFIX}_${tmp_size}_rep_${rep}"

    ${MPIEXEC} ${FLAGS_MPI_BATCH} -- \
    env LIKWID_EVENTS=${LIKWID_EVENTS} \
    env RESULT_PATH=${RESULT_PATH} \
    env OMP_NUM_THREADS=${OMP_NUM_THREADS} \
    env OMP_PLACES=${OMP_PLACES} \
    env OMP_PROC_BIND=${OMP_PROC_BIND} \
    env NUM_TASKS=${NUM_TASKS} \
    env LIKWID_FORCE=${LIKWID_FORCE} \
    env LIKWID_DEBUG=${LIKWID_DEBUG} \
    env CHAMELEON_TOOL_LIBRARIES=${CHAMELEON_TOOL_LIBRARIES} \
    ${MATRIX_EXE} ${tmp_size} ${NUM_TASKS} ${NUM_TASKS} ${NUM_TASKS} ${NUM_TASKS} &> ${RESULT_DIR}/result_${LOG_POSTFIX}_${tmp_size}_rep_${rep}.log
}

for tmp_ver in "${VERSIONS[@]}"
do
    export LOG_POSTFIX=${tmp_ver}
    export CHAMELEON_TOOL_LIBRARIES=${SCRIPT_DIR}/tool_${tmp_ver}.so

    for tmp_size in "${M_SIZES[@]}"
    do
        for rep in {1..${NUM_REP}}
        do
            echo "Running experiments for version ${tmp_ver} and matrix size ${tmp_size} - repetition ${rep}"
            run_experiment
        done
    done
done

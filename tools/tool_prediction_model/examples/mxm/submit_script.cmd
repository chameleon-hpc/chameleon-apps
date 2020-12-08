#!/bin/sh
#SBATCH -J toolpred_test
#SBATCH -o ./results/ch-mxm_test_tool_%J.out
#SBATCH -e ./results/ch-mxm_test_tool_%J.err
#SBATCH -D ./
#SBATCH --time=00:30:00
#SBATCH --get-user-env
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --account=pr58ci
#SBATCH --partition=test

module load slurm_setup

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export VT_LOGFILE_PREFIX=/dss/dsshome1/0A/di49mew/chameleon-scripts/cham_tool/tool_integ_predmodel/examples/mxm/traces
export CHAM_TOOL_DIR=/dss/dsshome1/0A/di49mew/chameleon-scripts/cham_tool/tool_integ_predmodel/build/libtool.so
export NUM_TASK=200

# Run the program with cham_tool
mpirun -n 2 ./mxm_chameleon $NUM_TASK $NUM_TASK

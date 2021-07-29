#!/usr/local_rwth/bin/zsh
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=chameleon_pingpong 
#SBATCH --output=chameleon_pingpong.%J.txt
#SBATCH --exclusive
#SBATCH --account=jara0001
#SBATCH --partition=c16m

# =============== Load desired modules
source /home/jk869269/.zshrc
source env_ch_intel.sh
module load DEV-TOOLS papi

# =============== Settings & environment variables
CUR_DATE_STR=${CUR_DATE_STR:-"$(date +"%Y%m%d_%H%M%S")"}

# Tests for Type 0
make
make run-sm-socket &> result_type0_socket.log
make run-sm-cache2 &> result_type0_cache2.log
make run-dm &> result_type0_dm.log

# Tests for Type 1
ADDITIONAL_COMPILE_FLAGS="-DBENCHMARK_TYPE=1" make
make run-sm-socket &> result_type1_socket.log
make run-sm-cache2 &> result_type1_cache2.log
make run-dm &> result_type1_dm.log

# Tests for Type 2
ADDITIONAL_COMPILE_FLAGS="-DBENCHMARK_TYPE=2" make
make run-sm-socket &> result_type2_socket.log
make run-sm-cache2 &> result_type2_cache2.log
make run-dm &> result_type2_dm.log

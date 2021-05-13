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
module load chameleon

make run-batch

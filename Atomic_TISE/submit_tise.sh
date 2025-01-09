#!/usr/bin/bash
#SBATCH --job-name TISE
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 32
#SBATCH --mem=144G
#SBATCH -o run.log 
#SBATCH -t 0-08:00:00

#SBATCH --exclude=node73
#SBATCH --exclude=node81

REPO_DIR="/users/becker/dopl4670/Research/Atomic_Schrodinger/Atomic_TISE"
                                                               
hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/tise_main.py >> results.log





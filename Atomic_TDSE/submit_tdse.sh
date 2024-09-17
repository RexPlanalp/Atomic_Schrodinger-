#!/usr/bin/bash
#SBATCH --job-name TDSE
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 32
#SBATCH --mem=32G
#SBATCH -o run.log 
#SBATCH -t 1-00:00:00

#SBATCH --exclude=node73

REPO_DIR="/users/becker/dopl4670/Research/Atomic_Schrodinger/Atomic_TDSE"
                                                               
hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/tdse_main.py >> results.log





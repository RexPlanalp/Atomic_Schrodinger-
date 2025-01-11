#!/usr/bin/bash
#SBATCH --job-name PES
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=128G
#SBATCH -o run.log 
#SBATCH -t 7-00:00:00

#SBATCH --exclude=node48

REPO_DIR="/users/becker/dopl4670/Research/Atomic_Schrodinger/Atomic_PES"
                                                               
hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/pes_main.py >> results.log





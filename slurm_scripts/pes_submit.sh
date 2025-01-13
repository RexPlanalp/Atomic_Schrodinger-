#!/usr/bin/bash
#SBATCH --job-name simulation
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=16G
#SBATCH -o run.log 
#SBATCH -t 1-00:00:00


#SBATCH --exclude=node41
#SBATCH --exclude=node48

REPO_DIR="/users/becker/dopl4670/Research/Atomic_Schrodinger/src"
                                                               
hostname
pwd



mpiexec -n $SLURM_NTASKS python $REPO_DIR/pes_main.py >> results.log





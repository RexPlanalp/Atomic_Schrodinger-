#!/usr/bin/bash
#SBATCH --job-name A
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=32G
#SBATCH -o run.log 
#SBATCH -t 0-02:00:00


REPO_DIR="/users/becker/dopl4670/Research/Atomic_Schrodinger/General_Plotting"
                                                             

hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/Asymmetry.py $1 $2 >> results.log





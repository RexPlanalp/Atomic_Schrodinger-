#!/usr/bin/bash
#SBATCH --job-name misc
#SBATCH -p jila
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem=128G
#SBATCH -o run.log 
#SBATCH -t 1-00:00:00


REPO_DIR="/users/becker/dopl4670/Research/Atomic_Schrodinger/Research_Plotting"
                                                             

hostname
pwd

mpiexec -n $SLURM_NTASKS python $REPO_DIR/A_amplitude_sweep.py $1 >> results.log



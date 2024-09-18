import numpy as np
import time
import h5py
import sys
import os
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_SELF
rank = comm.Get_rank()
size = comm.Get_size()

sys.path.append('/users/becker/dopl4670/Research/TDSE/Common')
from Sim import *

if rank == 0:
    if not os.path.exists("PES_files"):
        os.mkdir("PES_files")
comm.barrier()
    
if rank == 0:   
    sim_start = time.time()
    print("Setting up Simulation...")
    print("\n")
simInstance = Sim("input.json")  
simInstance.lm_block_maps() 
simInstance.calc_n_block()   
simInstance.timeGrid() 
simInstance.spacialGrid() 

n_basis = simInstance.splines["n_basis"]
order = simInstance.splines["order"]
lmax = simInstance.lm["lmax"]
n_block = simInstance.n_block
pot = simInstance.box["pot"]
lmax = simInstance.lm["lmax"]
    
total_size = n_basis * n_block
lm_dict,block_dict = simInstance.lm_dict,simInstance.block_dict
if rank == 0:   
    sim_end = time.time()
    print(f"Finished Setting up in {round(sim_end-sim_start,6)} seconds")
    print("\n")

if True:
    S = PETSc.Mat().createAIJ([total_size, total_size], nnz=(2 * (order - 1) + 1), comm=comm)
    viewer = PETSc.Viewer().createBinary('TISE_files/S.bin', 'r')
    S.load(viewer)
    viewer.destroy()

    Si, Sj, Sv = S.getValuesCSR()
    S.destroy()
    S_array = csr_matrix((Sv, Sj, Si))
    S_R= S_array[:n_basis, :n_basis]
    S.destroy()

if True:  
    with h5py.File('TDSE_files/TDSE.h5', 'r') as f:
        data = f["psi_final"][:]
        real_part = data[:, 0]
        imaginary_part = data[:, 1]
        wavefunction = real_part + 1j * imaginary_part

lm_list = []
pyramid = [[None for _ in range(2*lmax + 1)] for _ in range(lmax + 1)]


CONT = "CONT" in sys.argv

for (l,m),block_index in lm_dict.items():
    print(f"Computing Probability of having l,m = {l,m}")
    wavefunction_block = wavefunction[block_index*n_basis:(block_index+1)*n_basis]

    if CONT:
        with h5py.File(f'TISE_files/{pot}.h5', 'r') as f:
            datasets = list(f.keys())
            for dataset_name in datasets:
                if dataset_name.startswith('Psi_'):
                    parts = dataset_name.split('_')
                    current_n = int(parts[1])
                    current_l = int(parts[2])
                    
                    if current_l == l:
                       
                        data = f[dataset_name][:]
                        real_part = data[:, 0]
                        imaginary_part = data[:, 1]
                        bound_state = real_part + 1j * imaginary_part

                    
                        inner_product = bound_state.conj().dot(S_R.dot(wavefunction_block))
                        wavefunction_block -= inner_product * bound_state
                        
    probability = wavefunction_block.conj().dot(S_R.dot(wavefunction_block))

    pyramid[l][m + lmax] = np.real(probability)

    # Need to generalize, only true for RHS circular at the moment
    if l == m:
        lm_list.append(np.real(probability))


plt.figure()
l_array = [l for l,m in lm_dict.keys() if l == m]
plt.bar(l_array,lm_list,color = "k")
plt.yscale('log')
if CONT:
    plt.title("Cont States")
    
else:
    plt.title("Bound + Cont States")
print("#######################")
print("Total Probability",np.sum(lm_list))
plt.savefig("images/blocks.png")
plt.xlabel("Block")
plt.ylabel("Probability")
plt.clf()
##########
pyramid_array = np.array([[val if val is not None else 0 for val in row] for row in pyramid])

# Plotting the pyramid as a heatmap
pyramid_array = np.array([[val if val is not None else 0 for val in row] for row in pyramid])

# Plotting the pyramid as a heatmap
fig, ax = plt.subplots(figsize=(10, 8))
#cax = ax.imshow(pyramid_array[::-1], cmap='hot', interpolation='nearest')  # Reverse the array for upside-down pyramid
cax = ax.imshow(pyramid_array[::-1], cmap='inferno', interpolation='nearest',norm = LogNorm())  # Reverse the array for upside-down pyramid
ax.set_xlabel('m')
ax.set_ylabel('l')

fig.colorbar(cax, ax=ax,shrink = 0.5)
plt.title('Heatmap of Probabilities for l and m Values')
plt.savefig("images/pyramid.png")
plt.show()









        
    
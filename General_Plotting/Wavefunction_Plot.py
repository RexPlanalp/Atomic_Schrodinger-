import numpy as np
import sys
import h5py 
import matplotlib.pyplot as plt


sys.path.append('/users/becker/dopl4670/Research/TDSE/Common')
from Sim import *
from Basis import *

simInstance = Sim("input.json")  
simInstance.lm_block_maps() 
simInstance.calc_n_block()   
simInstance.timeGrid() 
simInstance.spacialGrid() 

basisInstance = basis()
basisInstance.createKnots(simInstance)

order = simInstance.splines["order"]
knots = basisInstance.knots


if "BOUND" in sys.argv:
    potential = simInstance.box["pot"]
    n = int(sys.argv[2])
    l = int(sys.argv[3])

    with h5py.File(f'TISE_files/{potential}.h5', 'r') as f:
        data = f[f"/Psi_{n}_{l}"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
        total = real_part + 1j*imaginary_part

if "TDSE" in sys.argv:
    l = int(sys.argv[2])
    m = int(sys.argv[3])
    block = simInstance.lm_dict[(l,m)]
    n_basis = simInstance.splines["n_basis"]
    with h5py.File('TDSE_files/TDSE.h5', 'r') as f:
        data = f["psi_final"][:]
        real_part = data[:, 0]
        imaginary_part = data[:, 1]
        total = real_part + 1j * imaginary_part
    total = total[block*n_basis:(block+1)*n_basis]


x = np.linspace(0,simInstance.box["grid_size"],15000)
wavefunction = np.zeros_like(x,dtype=complex)

for i in range(simInstance.splines["n_basis"]):
    start = knots[i]
    end = knots[i + order]
    
    valid_indices = np.where((x >= start) & (x < end))[0]
    
    if valid_indices.size > 0:
        wavefunction[valid_indices] += total[i] * basisInstance.B(i, order, x[valid_indices], knots)



plt.plot(x,np.abs(wavefunction)**2)
plt.xlim([0,30])
plt.savefig("images/PDF.png")

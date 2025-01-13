import h5py as h5py
import os as os
import pickle as pickle
import json as json
import simulation_class as simulation
import sys as sys

import matplotlib.pyplot as plt
import numpy as np
import pyshtools as pysh
from matplotlib.colors import LogNorm

from scipy.special import sph_harm
from scipy.integrate import simpson
from scipy.sparse import csr_matrix


from petsc4py import PETSc
from slepc4py import SLEPc

comm = PETSc.COMM_WORLD
rank = comm.rank

class block(simulation.simulation):
    def __init__(self,input_file):
        super().__init__(input_file)

    def loadS(self):
        S = PETSc.Mat().createAIJ(self.input_params["splines"]["n_basis"],self.input_params["splines"]["n_basis"] , nnz=(2 * (self.input_params["splines"]["order"] - 1) + 1), comm=PETSc.COMM_SELF)
        viewer = PETSc.Viewer().createBinary('TISE_files/S.bin', 'r')
        S.load(viewer)
        viewer.destroy()
        

        Si, Sj, Sv = S.getValuesCSR()
        S_array = csr_matrix((Sv, Sj, Si))
        self.S = S_array
        S.destroy()
        return True

    def load_final_state(self):
        with h5py.File('TDSE_files/TDSE.h5', 'r') as f:
            tdse_data = f["psi_final"][:]
            real_part = tdse_data[:, 0]
            imaginary_part = tdse_data[:, 1]
            psi_final_bspline = real_part + 1j * imaginary_part
        self.final_state = psi_final_bspline

    def compute_norm(self,CONT):
        print("Computing Total Norm...")
        print('\n')
 
        total = 0
        for (l, m), block_index in self.input_params["lm"]["lm_to_block"].items():
            wavefunction_block = self.final_state[block_index * self.input_params["splines"]["n_basis"]:(block_index + 1) * self.input_params["splines"]["n_basis"]]
            if CONT:
                wavefunction_block = self.project_out_bound_states(wavefunction_block,l)
            norm = wavefunction_block.conj().dot(self.S.dot(wavefunction_block))
            total += norm
        print("Total Norm:", np.real(total))
        print('\n')

        return np.real(total)

    def project_out_bound_states(self,wavefunction_block,l):
        with h5py.File(f'TISE_files/{self.input_params["species"]}.h5', 'r') as f:
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

                                # Project out the bound state
                                inner_product = bound_state.conj().dot(self.S.dot(wavefunction_block))
                                wavefunction_block -= inner_product * bound_state

        return wavefunction_block
    
    def compute_distribution(self, projOutBound):
        pyramid = [[None for _ in range(2 * self.input_params["lm"]["lmax"] + 1)] for _ in range(self.input_params["lm"]["lmax"] + 1)]

        # Loop over l and m values, computing probabilities
        for (l, m), block_index in self.input_params["lm"]["lm_to_block"].items():
            
            wavefunction_block = self.final_state[block_index * self.input_params["splines"]["n_basis"]:(block_index + 1) * self.input_params["splines"]["n_basis"]]

            # If CONT flag is set, project out bound states
            if projOutBound:
                wavefunction_block = self.project_out_bound_states(wavefunction_block,l)

            # Compute probability
            probability = wavefunction_block.conj().dot(self.S.dot(wavefunction_block))
            pyramid[l][m + self.input_params["lm"]["lmax"]] = np.real(probability)

            # Add to lm_list if l == m (for circular polarization)
            
        # Generate the pyramid heatmap
        pyramid_array = np.array([[val if val is not None else 0 for val in row] for row in pyramid])
        pyramid_array[pyramid_array==0] = np.min(pyramid_array[pyramid_array!=0])
        np.save("TDSE_files/pyramid_array.npy",pyramid_array)
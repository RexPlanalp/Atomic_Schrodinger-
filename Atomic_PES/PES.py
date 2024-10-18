import matplotlib.pyplot as plt
import numpy as np

import h5py
import sys
import time
import os
import pickle
import json
import gc

from numpy import exp
from numpy import sqrt

from scipy.special import sph_harm
from scipy.integrate import simps
from scipy.sparse import csr_matrix

sys.path.append('/users/becker/dopl4670/Research/Atomic_Schrodinger/Common')
import utility

from petsc4py import PETSc
from slepc4py import SLEPc

comm = PETSc.COMM_WORLD
rank = comm.rank

class PES:
    def __init__(self, input_file):
        with open(input_file, 'r') as file:
            self.parameters = json.load(file)
        self.input_file = input_file

     


        polarization = self.parameters["lasers"]["polarization"]
        polarization /= np.linalg.norm(polarization)

        poynting = self.parameters["lasers"]["poynting"]
        poynting /= np.linalg.norm(poynting)

        ellipticity_Vector = np.cross(polarization, poynting) 
        ellipticity_Vector /= np.linalg.norm(ellipticity_Vector)

        ell = self.parameters["lasers"]["ell"]
        
        components = [(1 if polarization[i] != 0 or ell * ellipticity_Vector[i] != 0 else 0) for i in range(3)]

        
        self.lm_dict,self.block_dict = utility.lm_expansion(self.parameters["lm"]["lmax"],self.parameters["lm"]["mmin"],self.parameters["lm"]["mmax"],self.parameters["lm"]["expansion"], \
                                                            self.parameters["state"], \
                                                                components)
        self.parameters["total_size"] = len(self.lm_dict) * self.parameters["splines"]["n_basis"]
        self.parameters["n_block"] = len(self.lm_dict)

        def H(x):
            return (-1/(x+1E-25))          
        def He(x):
                return (-1/sqrt(x**2 + 1E-25)) + -1.0*exp(-2.0329*x)/sqrt(x**2 + 1E-25)  - 0.3953*exp(-6.1805*x)
        def Ar(x):
            return -1.0 / (x + 1e-25) - 17.0 * exp(-0.8103 * x) / (x + 1e-25) \
            - (-15.9583) * exp(-1.2305 * x) \
            - (-27.7467) * exp(-4.3946 * x) \
            - 2.1768 * exp(-86.7179 * x)
        
        if self.parameters["species"] == "H":
            self.pot_func = H
        elif self.parameters["species"] == "He":
            self.pot_func = He
        elif self.parameters["species"] == "Ar":
            self.pot_func = Ar

        rmax = self.parameters["box"]["grid_size"]
        dr = self.parameters["box"]["grid_spacing"]
        self.r_range = np.arange(dr, rmax+dr, dr)
        self.Nr = len(self.r_range)

        dE,Emax = self.parameters["E"]
        self.E_range = np.round(np.arange(dE, Emax+dE, dE),decimals=3)
        
    def get(self, key, default=None):
        return self.parameters.get(key, default)
    
    def __getitem__(self, key):
        return self.parameters[key]
    
    def __setitem__(self, key, value):
        self.parameters[key] = value
    
    def __repr__(self):
        return f"PES({self.input_file})"
    
    def loadS_R(self):

        n_basis = self.parameters["splines"]["n_basis"]
        order = self.parameters["splines"]["order"]
         
        S = PETSc.Mat().createAIJ([self.parameters["total_size"], self.parameters["total_size"]], nnz=(2 * (order - 1) + 1), comm=PETSc.COMM_SELF)
        viewer = PETSc.Viewer().createBinary('TISE_files/S.bin', 'r')
        S.load(viewer)
        viewer.destroy()

        from scipy.sparse import csr_matrix
        Si, Sj, Sv = S.getValuesCSR()
        S.destroy()
        S_array = csr_matrix((Sv, Sj, Si))
        self.S_R= S_array[:n_basis, :n_basis]
        S.destroy()

    def project_out_bound_states(self):
        """
        Projects out bound states from psi_final_bspline for each (l, m) channel.
        
        Parameters:
        - psi_final_bspline: The input wavefunction in the B-spline basis.
        - lm_dict: Dictionary containing (l, m) as keys and block indices as values.
        - n_basis: The number of basis functions.
        - pot: The potential used, for accessing the corresponding bound state data.
        - S_R: The overlap matrix in the reduced basis.
        
        Returns:
        - psi_final_bspline with the bound states projected out.
        """
        self.final_state_cont = np.zeros_like(self.final_state)

        n_basis = self.parameters["splines"]["n_basis"]
        potential = self.parameters["species"]

        for (l, m),block_idx in self.lm_dict.items():
            wavefunction_block = self.final_state[block_idx * n_basis:(block_idx + 1) * n_basis]

            # Load the bound states from the HDF5 file corresponding to the potential
            with h5py.File(f'TISE_files/{potential}.h5', 'r') as f:
                datasets = list(f.keys())
                for dataset_name in datasets:
                    if dataset_name.startswith('Psi_'):
                        parts = dataset_name.split('_')
                        current_n = int(parts[1])
                        current_l = int(parts[2])

                        # Check if the current bound state corresponds to the current l value
                        if current_l == l:
                            data = f[dataset_name][:]
                            real_part = data[:, 0]
                            imaginary_part = data[:, 1]
                            bound_state = real_part + 1j * imaginary_part

                            # Project out the bound state
                            inner_product = bound_state.conj().dot(self.S_R.dot(wavefunction_block))
                            wavefunction_block -= inner_product * bound_state

            # Update the block in psi_final_bspline with the modified wavefunction
            self.final_state_cont[block_idx * n_basis:(block_idx + 1) * n_basis] = wavefunction_block

    def compute_coulomb_waves_and_phases(self,E):
        r_range2 = self.r_range ** 2
        dr = self.r_range[1] - self.r_range[0]
        dr2 = dr * dr

        lmax = self.parameters["lm"]["lmax"]
        # Assuming l values range from 0 to lmax
        l_values = np.arange(0, lmax + 1)
        l_term = l_values * (l_values + 1)  # Shape (num_l,)
        k = np.sqrt(2 * E)

        # Prepare potential array, replicate for each l
        potential = np.empty_like(self.r_range)
        potential[0] = np.inf  # Set the potential at r=0 to avoid division by zero.
        potential[1:] = self.pot_func(self.r_range[1:])

        # Vectorize potential to shape (num_l, len(r_range))
        potential = np.broadcast_to(potential, (l_values.size, self.r_range.size))

        # Initialize Coulomb wave array for all l values
        coul_wave = np.zeros((l_values.size, self.r_range.size))
        coul_wave[:, 0] = self.r_range[0]** (l_values + 1)  # Initialize first point for all l values

        # Adjust initialization for the second point
        coul_wave[:, 1] = coul_wave[:, 0] * (dr2 * (l_term / r_range2[1] + 2 * potential[:, 1] + 2 * E) + 2)

        # Compute wave function values in a vectorized manner
        for idx in range(2, len(self.r_range)):
            term = dr2 * (l_term / r_range2[idx - 1] + 2 * potential[:, idx - 1] - 2 * E)
            coul_wave[:, idx] = coul_wave[:, idx - 1] * (term + 2) - coul_wave[:, idx - 2]

        # Final values and phase computation
        r_val = self.r_range[-2]
        coul_wave_r = coul_wave[:, -2]
        dcoul_wave_r = (coul_wave[:, -1] - coul_wave[:, -3]) / (2 * dr)

        norm = np.sqrt(np.abs(coul_wave_r) ** 2 + (np.abs(dcoul_wave_r) / (k + 1 / (k * r_val))) ** 2)
        phase = np.angle((1.j * coul_wave_r + dcoul_wave_r / (k + 1 / (k * r_val))) /
                        (2 * k * r_val) ** (1.j / k)) - k * r_val + l_values * np.pi / 2

        # Normalize Coulomb wave functions
        coul_wave /= norm[:, np.newaxis]

        return phase, coul_wave

    def load_final_state(self):
        with h5py.File('TDSE_files/TDSE.h5', 'r') as f:
            tdse_data = f["psi_final"][:]
            real_part = tdse_data[:, 0]
            imaginary_part = tdse_data[:, 1]
            psi_final_bspline = real_part + 1j * imaginary_part
        self.final_state = psi_final_bspline

    def expand_final_state(self,basisInstance):
        n_block = self.parameters["n_block"]
        n_basis = self.parameters["splines"]["n_basis"]
        order = self.parameters["splines"]["order"]
        knots = basisInstance.knots


        self.final_state_cont_pos = np.zeros(self.Nr*n_block, dtype=np.complex128)
        for (l,m),block_idx in self.lm_dict.items():
            block = self.final_state_cont[block_idx*n_basis:(block_idx+1)*n_basis]
            wavefunction = np.zeros_like(self.r_range,dtype=complex)

            for i in range(n_basis):
                
                start = knots[i]
                end = knots[i + order]
                
                
                valid_indices = np.where((self.r_range >= start) & (self.r_range < end))[0]
                
                if valid_indices.size > 0:
                    
                    wavefunction[valid_indices] += block[i] * basisInstance.B(i, order, self.r_range[valid_indices], knots)
            start_idx = block_idx * len(self.r_range)
            end_idx = (block_idx + 1) * len(self.r_range)
            
        
            self.final_state_cont_pos[start_idx:end_idx] = wavefunction

    def compute_partial_spectra(self):
        self.partial_spectra = {}
        self.phases = {}

        # Initialize the partial spectra dictionary for each (l, m)
        for (l, m) in self.lm_dict.keys():
            self.partial_spectra[(l, m)] = []

        for E in self.E_range:
            print(E)
            phases, waves = self.compute_coulomb_waves_and_phases(E)
            for (l, m), block_idx in self.lm_dict.items():
                block = self.final_state_cont_pos[block_idx * self.Nr:(block_idx + 1) * self.Nr]
                y = waves[l].conj() * block
                inner_product = simps(y, self.r_range)
                # Append the inner product to the list for each (l, m)
                self.partial_spectra[(l, m)].append(inner_product)
                if (E, l) not in self.phases:
                    self.phases[(E, l)] = phases[l]
            # Delete waves and phases to free up memory
            del waves, phases

        with open('PES_files/partial_spectra.pkl', 'wb') as pickle_file:
            pickle.dump(self.partial_spectra, pickle_file)
        with open('PES_files/phases.pkl', 'wb') as pickle_file:
            pickle.dump(self.phases, pickle_file)

    def compute_angle_integrated(self):
        PES = 0
        for key, value in self.partial_spectra.items():
            PES += np.abs(value)**2
        PES /= (2*np.pi)**3

        np.save("PES_files/PES.npy", PES)
        np.save("PES_files/E_range.npy", self.E_range)

        return None
    
    def compute_angle_resolved(self):
  

        k_range = np.sqrt(2 * self.E_range)

        if self.parameters["SLICE"] == "XZ":
            theta_range = np.arange(0, np.pi, 0.01)
            phi_range = np.array([0, np.pi])

        elif self.parameters["SLICE"] == "XY":
            theta_range = np.array([np.pi / 2])
            phi_range = np.arange(0, 2 * np.pi, 0.01)

        # Create theta and phi grids
        theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing='ij')
        # Flatten the grids for vectorized computation
        theta_vals = theta_grid.flatten()
        phi_vals = phi_grid.flatten()

        # Precompute sph_harm for all combinations of l, m over the theta and phi grids
        sph_harm_values = {}
        for (l, m), _ in self.partial_spectra.items():
            # sph_harm takes vectorized phi and theta
            sph_harm_values[(l, m)] = sph_harm(m, l, phi_vals, theta_vals)

        # Number of k values and grid points
        N_k = len(k_range)
        N_points = len(theta_vals)

        # Initialize arrays to store the results
        pad_vals = np.zeros((N_k, N_points))
        k_vals = np.repeat(k_range[:, np.newaxis], N_points, axis=1)  # Shape (N_k, N_points)

        # Loop over k values
        for i, k in enumerate(k_range):
            E = self.E_range[i]
            print(f"Processing energy: {E}")
            if k == 0:
                continue
            E_idx = i  # Assuming k_range and E_range are aligned

            # Initialize pad_amp as a zero array with the same length as theta_vals
            pad_amp = np.zeros(N_points, dtype=complex)

            # Sum over all l, m contributions
            for (l, m), value in self.partial_spectra.items():
                sph = sph_harm_values[(l, m)]
                phase_factor = (-1j) ** l * np.exp(1j * self.phases[(E, l)]) * value[E_idx]
                pad_amp += phase_factor * sph

            # Compute the squared magnitude of pad_amp
            pad_vals[i, :] = np.abs(pad_amp) ** 2

        # Adjust pad_vals
        pad_vals /= (2 * np.pi) ** 3
        pad_vals /= k_vals

        # Flatten arrays for saving
        k_vals_flat = k_vals.flatten()
        pad_vals_flat = pad_vals.flatten()
        theta_vals_repeated = np.tile(theta_vals, N_k)
        phi_vals_repeated = np.tile(phi_vals, N_k)

        # Stack and save the data
        PAD = np.vstack((k_vals_flat, theta_vals_repeated, phi_vals_repeated, pad_vals_flat))
        np.save("PES_files/PAD.npy", PAD)

        return None


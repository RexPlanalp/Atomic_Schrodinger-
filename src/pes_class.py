import h5py as h5py
import os as os
import pickle as pickle
import json as json
import simulation_class as simulation

import matplotlib.pyplot as plt
import numpy as np
import pyshtools as pysh

from scipy.special import sph_harm
from scipy.integrate import simpson
from scipy.sparse import csr_matrix


from petsc4py import PETSc
from slepc4py import SLEPc

comm = PETSc.COMM_WORLD
rank = comm.rank

class pes(simulation.simulation):
    def __init__(self,input_file):
        super().__init__(input_file)

        r_start,r_end,Nr = self.input_params["box"]["r_range"]
        self.r_range = np.linspace(r_start,r_end,int(Nr))

        dE,Emax = self.input_params["E"]
        self.E_range = np.round(np.arange(dE, Emax+dE, dE),decimals=3)
        
    def ylm(self,l,m,theta,phi):
        return pysh.expand.spharm_lm(l,m,theta,phi, kind = 'complex', degrees = False,csphase = -1,normalization = "ortho")

    def loadS(self):

         
        S = PETSc.Mat().createAIJ(self.input_params["splines"]["n_basis"] ,self.input_params["splines"]["n_basis"], nnz=(2 * (self.input_params["splines"]["order"] - 1) + 1), comm=PETSc.COMM_SELF)
        viewer = PETSc.Viewer().createBinary('TISE_files/S.bin', 'r')
        S.load(viewer)
        viewer.destroy()

        
        Si, Sj, Sv = S.getValuesCSR()
        S.destroy()
        S_array = csr_matrix((Sv, Sj, Si))
        self.S_R = S_array
      
    def project_out_bound_states(self):
        self.final_state_cont = np.zeros_like(self.final_state)

        for (l, m),block_idx in self.input_params["lm"]["lm_to_block"].items():
            wavefunction_block = self.final_state[block_idx * self.input_params["splines"]["n_basis"]:(block_idx + 1) * self.input_params["splines"]["n_basis"]]

            # Load the bound states from the HDF5 file corresponding to the potential
            with h5py.File(f'TISE_files/{self.input_params["species"]}.h5', 'r') as f:
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
            self.final_state_cont[block_idx * self.input_params["splines"]["n_basis"]:(block_idx + 1) * self.input_params["splines"]["n_basis"]] = wavefunction_block

    def compute_coulomb_waves_and_phases(self, E):
        r_range2 = self.r_range ** 2
        dr = self.r_range[1] - self.r_range[0]
        dr2 = dr * dr

        lmax = self.input_params["lm"]["lmax"]
        l_values = np.arange(0, lmax + 1)
        l_term = l_values * (l_values + 1)  # Shape (num_l,)
        k = np.sqrt(2 * E)

        # Prepare potential array and replicate for each l
        potential = np.empty_like(self.r_range)
        potential[0] = np.inf  # Avoid division by zero at r=0
        potential[1:] = self.input_params["box"]["potential_func"](self.r_range[1:])
        potential = np.broadcast_to(potential, (l_values.size, self.r_range.size))

        # Initialize Coulomb wave array for all l values
        coul_wave = np.zeros((l_values.size, self.r_range.size))
        coul_wave[:, 0] = 1  # Initialize first point for all l values

        # Adjust initialization for the second point
        coul_wave[:, 1] = coul_wave[:, 0] * (
            dr2 * (l_term / r_range2[1] + 2 * potential[:, 1] + 2 * E) + 2
        )

        # Compute wave function values in a vectorized manner
        for idx in range(2, len(self.r_range)):
            term = dr2 * (
                l_term / r_range2[idx - 1] + 2 * potential[:, idx - 1] - 2 * E
            )
            coul_wave[:, idx] = (
                coul_wave[:, idx - 1] * (term + 2) - coul_wave[:, idx - 2]
            )

            # Overflow catch for each l value
            overflow_indices = np.abs(coul_wave[:, idx]) > 1E10
            if np.any(overflow_indices):
                coul_wave[overflow_indices] /= np.max(np.abs(coul_wave[overflow_indices, idx]))

        # Final values and phase computation
        r_val = self.r_range[-2]
        coul_wave_r = coul_wave[:, -2]
        dcoul_wave_r = (coul_wave[:, -1] - coul_wave[:, -3]) / (2 * dr)

        norm = np.sqrt(
            np.abs(coul_wave_r) ** 2
            + (np.abs(dcoul_wave_r) / (k + 1 / (k * r_val))) ** 2
        )
        phase = (
            np.angle(
                (
                    1j * coul_wave_r + dcoul_wave_r / (k + 1 / (k * r_val))
                )
                / (2 * k * r_val) ** (1j / k)
            )
            - k * r_val
            + l_values * np.pi / 2
        )

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

    def expand_final_state(self, basis):
        

        # Initialize the final state array
        self.final_state_cont_pos = np.zeros(len(self.r_range) * self.input_params["lm"]["n_blocks"], dtype=np.complex128)

        # Loop over each B-spline index `i`
        for i in range(self.input_params["splines"]["n_basis"]):
            start = basis.knots[i]
            end = basis.knots[i + self.input_params["splines"]["order"]]

            # Find the indices where the B-spline is non-zero
            valid_indices = np.where((self.r_range >= start) & (self.r_range < end))[0]
            if valid_indices.size > 0:
                # Evaluate the B-spline once for these indices
                y = basis.B(i, self.input_params["splines"]["order"], self.r_range[valid_indices], basis.knots)

                # Loop over each block `(l, m)`
                for (l, m), block_idx in self.input_params["lm"]["lm_to_block"].items():
                    # Extract the coefficient for this B-spline in the current block
                    block = self.final_state_cont[block_idx * self.input_params["splines"]["n_basis"]:(block_idx + 1) * self.input_params["splines"]["n_basis"]]
                    coef = block[i]

                    # Compute the starting index in the flattened array
                    start_idx = block_idx * len(self.r_range)

                    # Update the wavefunction for this block
                    self.final_state_cont_pos[start_idx + valid_indices] += coef * y

    def compute_partial_spectra(self):
        if os.path.exists('PES_files/partial_spectra.pkl'):
            return
         
         


        self.partial_spectra = {}
        self.phases = {}

        # Initialize the partial spectra dictionary for each (l, m)
        for (l, m) in self.input_params["lm"]["lm_to_block"].keys():
            self.partial_spectra[(l, m)] = []

        for E in self.E_range:
            print(E)
            phases, waves = self.compute_coulomb_waves_and_phases(E)
            for (l, m), block_idx in self.input_params["lm"]["lm_to_block"].items():
                block = self.final_state_cont_pos[block_idx * len(self.r_range):(block_idx + 1) * len(self.r_range)]
                y = waves[l].conj() * block
                inner_product = simpson(y=y, x=self.r_range)
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
        if os.path.exists('PES_files/partial_spectra.pkl'):
            with open('PES_files/partial_spectra.pkl', 'rb') as file:
                partial_spectra = pickle.load(file)
        else:
            partial_spectra = self.partial_spectra


        PES = 0
        for key, value in partial_spectra.items():
            PES += np.abs(value)**2
        PES /= (2*np.pi)**3

        np.save("PES_files/PES.npy", PES)
        np.save("PES_files/E_range.npy", self.E_range)

        return None
    
    def compute_angle_resolved(self):

        if os.path.exists('PES_files/partial_spectra.pkl'):
            with open('PES_files/partial_spectra.pkl', 'rb') as file:
                partial_spectra = pickle.load(file)
            with open('PES_files/phases.pkl', 'rb') as file:
                phases = pickle.load(file)
        else:
            partial_spectra = self.partial_spectra
  

        k_range = np.sqrt(2 * self.E_range)

        if self.input_params["SLICE"] == "XZ":
            theta_range = np.arange(0, np.pi, 0.01)
            phi_range = np.array([0, np.pi])

        elif self.input_params["SLICE"] == "XY":
            theta_range = np.array([np.pi / 2])
            phi_range = np.arange(0, 2 * np.pi, 0.01)

        # Create theta and phi grids
        theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing='ij')
        # Flatten the grids for vectorized computation
        theta_vals = theta_grid.flatten()
        phi_vals = phi_grid.flatten()

        # Precompute sph_harm for all combinations of l, m over the theta and phi grids
        sph_harm_values = {}
        for (l, m), _ in partial_spectra.items():
            # sph_harm takes vectorized phi and theta
            sph_harm_values[(l, m)] = self.ylm(l, m, theta_vals, phi_vals)

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
            for (l, m), value in partial_spectra.items():
                sph = sph_harm_values[(l, m)]
                phase_factor = (-1j) ** l * np.exp(1j * phases[(E, l)]) * value[E_idx]
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


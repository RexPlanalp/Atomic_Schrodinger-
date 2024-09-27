import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import h5py
import sys
import json




sys.path.append('/users/becker/dopl4670/Research/Atomic_Schrodinger/Common')
import utility

from petsc4py import PETSc
from slepc4py import SLEPc

comm = PETSc.COMM_WORLD
rank = comm.rank

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
from scipy.sparse import csr_matrix
from petsc4py import PETSc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
from scipy.sparse import csr_matrix
from petsc4py import PETSc
import os

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

        self.lm_dict, self.block_dict = utility.lm_expansion(self.parameters["lm"]["lmax"],
                                                             self.parameters["state"],
                                                             components)
        self.parameters["total_size"] = len(self.lm_dict) * self.parameters["splines"]["n_basis"]
        self.parameters["n_block"] = len(self.lm_dict)

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

        Si, Sj, Sv = S.getValuesCSR()
        S_array = csr_matrix((Sv, Sj, Si))
        self.S_R = S_array[:n_basis, :n_basis]
        S.destroy()

    def load_final_state(self):
        with h5py.File('TDSE_files/TDSE.h5', 'r') as f:
            tdse_data = f["psi_final"][:]
            real_part = tdse_data[:, 0]
            imaginary_part = tdse_data[:, 1]
            psi_final_bspline = real_part + 1j * imaginary_part
        self.final_state = psi_final_bspline

    def compute_norm(self):
        print("Computing Total Norm...")
        print('\n')
        n_basis = self.parameters["splines"]["n_basis"]
        lmax = self.parameters["lm"]["lmax"]
        pot = self.parameters["species"]
        wavefunction = self.final_state
        lm_dict = self.lm_dict
        S_R = self.S_R

        total = 0
        for (l, m), block_index in lm_dict.items():
            wavefunction_block = wavefunction[block_index * n_basis:(block_index + 1) * n_basis]
            norm = wavefunction_block.conj().dot(S_R.dot(wavefunction_block))
            total += norm
        print("Total Norm:", np.real(total))
        print('\n')

        return np.real(total)

    def plot_distribution_slice(self,projOutCont,log):
        n_basis = self.parameters["splines"]["n_basis"]
        lmax = self.parameters["lm"]["lmax"]
        pot = self.parameters["species"]
        wavefunction = self.final_state
        lm_dict = self.lm_dict
        S_R = self.S_R

        probs = []
        ls = []
        for (l, m), block_index in lm_dict.items():
            if l != -m:
                continue
            wavefunction_block = wavefunction[block_index * n_basis:(block_index + 1) * n_basis]
            if projOutCont:
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

                                # Project out the bound state
                                inner_product = bound_state.conj().dot(S_R.dot(wavefunction_block))
                                wavefunction_block -= inner_product * bound_state



            norm = wavefunction_block.conj().dot(S_R.dot(wavefunction_block))
            probs.append(np.real(norm))
            ls.append(l)
        
        plt.bar(ls,probs,color = "k")
        if log:
            plt.yscale('log')
        else:
            plt.yscale('linear')
        plt.savefig("images/dist_slice.png")
        return None
    
    def compute_distribution(self, projOutCont,log):
        # Initialize variables
        n_basis = self.parameters["splines"]["n_basis"]
        lmax = self.parameters["lm"]["lmax"]
        pot = self.parameters["species"]
        wavefunction = self.final_state
        lm_dict = self.lm_dict
        S_R = self.S_R

        
        pyramid = [[None for _ in range(2 * lmax + 1)] for _ in range(lmax + 1)]

        # Loop over l and m values, computing probabilities
        for (l, m), block_index in lm_dict.items():
            
            wavefunction_block = wavefunction[block_index * n_basis:(block_index + 1) * n_basis]

            # If CONT flag is set, project out bound states
            if projOutCont:
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

                                # Project out the bound state
                                inner_product = bound_state.conj().dot(S_R.dot(wavefunction_block))
                                wavefunction_block -= inner_product * bound_state

            # Compute probability
            probability = wavefunction_block.conj().dot(S_R.dot(wavefunction_block))
            pyramid[l][m + lmax] = np.real(probability)

            # Add to lm_list if l == m (for circular polarization)
            
        # Generate the pyramid heatmap
        pyramid_array = np.array([[val if val is not None else 0 for val in row] for row in pyramid])
        fig, ax = plt.subplots(figsize=(10, 8))

        pyramid_array[pyramid_array==0] = np.min(pyramid_array[pyramid_array!=0])
        if log:
            cax = ax.imshow(pyramid_array[::-1], cmap='inferno', interpolation='nearest', norm=LogNorm())  # Reverse for upside-down pyramid
        else:
            cax = ax.imshow(pyramid_array[::-1], cmap='inferno', interpolation='nearest')
        ax.set_xlabel('m')
        ax.set_ylabel('l')

        fig.colorbar(cax, ax=ax, shrink=0.5)
        plt.title('Heatmap of Probabilities for l and m Values')
        plt.savefig("images/pyramid.png")
        plt.show()

if __name__ == '__main__':
    pes = PES("input.json")
    pes.loadS_R()
    pes.load_final_state()
    pes.compute_norm()
    pes.compute_distribution(projOutCont=False,log = False)
    pes.plot_distribution_slice(projOutCont=False,log = True)
    
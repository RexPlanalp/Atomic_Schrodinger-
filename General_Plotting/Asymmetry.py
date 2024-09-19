import numpy as np
import matplotlib.pyplot as plt

import pickle

from scipy.special import sph_harm
import matplotlib.colors as mcolors
from tqdm import tqdm
import sys

# Load partial spectra and phases
with open('PES_files/partial_spectra.pkl', 'rb') as file:
    partial_spectra = pickle.load(file)
with open('PES_files/phases.pkl', 'rb') as file:
    phases = pickle.load(file)


# Load energy range and compute momentum range
E_range = np.load("PES_files/E_range.npy")
k_range = np.sqrt(2 * E_range)


# Define parameters, depends on polarization
theta = np.pi / 2  # Fixed theta = pi/2
phi_range = np.arange(0, 2 * np.pi, 0.01)  # Range of phi values

if "SLICE" in sys.argv:

    E_target = float(sys.argv[2])  # The target energy
    theta = np.pi / 2  # Fixed theta = pi/2
    phi_range = np.arange(0, 2 * np.pi, 0.01)  # Range of phi values
    lm_vals = [(26,26),(25,25),(24,24),(23,23),(22,22),(21,21)]


    # Find the closest energy index to E = 0.48
    E_idx = np.argmin(np.abs(E_range - E_target))
    k = k_range[E_idx]  # The corresponding momentum value
    E = E_range[E_idx]  # The corresponding energy value

    # Initialize lists for phi values and asymmetry values
    phi_vals = []
    A_vals = []

    total_amplitues = {}
    total_phases = {}

    for key, value in partial_spectra.items():
        l, m = key
        if (l, m) not in lm_vals:
            continue
            
        total_amplitues[(l, m)] = np.abs((-1j) ** l * np.exp(1j * phases[(E, l)]) * value[E_idx])
        total_phases[(l, m)] = np.angle((-1j) ** l * np.exp(1j * phases[(E, l)]) * value[E_idx])

    # Loop over phi to compute the asymmetry
    for phi in tqdm(phi_range):

        phi_vals.append(phi)
        
        pad_amp = 0
        pad_amp_opp = 0
        
        # Compute partial wave expansions for forward and opposite directions
        for key, value in partial_spectra.items():
            l, m = key
            if (l, m) not in lm_vals:
                continue
            
        
            pad_amp += total_amplitues[key] * np.exp(1j*total_phases[key]) * sph_harm(m, l, phi, np.pi / 2) 
            pad_amp_opp += total_amplitues[key] * np.exp(1j*total_phases[key]) * sph_harm(m, l, phi+np.pi, np.pi / 2) 
        
        # Compute the photoelectron angular distributions
        pad_val = np.abs(pad_amp) ** 2
        pad_val_opp = np.abs(pad_amp_opp) ** 2
        
        # Calculate asymmetry A = (PAD - PAD_opp) / (PAD + PAD_opp)
        A = (pad_val - pad_val_opp) / (pad_val + pad_val_opp)
        A_vals.append(A)


    # Convert lists to numpy arrays
    phi_vals = np.array(phi_vals)
    A_vals = np.array(A_vals)


    # Plot phi vs asymmetry (Original Data)
    plt.plot(phi_vals, A_vals, label=f'Asymmetry at E = {E_target} a.u.', color='b')
    plt.xlabel('Phi (radians)')
    plt.ylabel('Asymmetry (A)')
    plt.title(f'Asymmetry vs Phi at E = {E_target} a.u.')
    plt.legend()
    plt.savefig("images/A_slice.png")
    plt.clf()

if "TOTAL" in sys.argv:
    PAD = np.load("PES_files/PAD.npy")

    k_vals = np.real(PAD[0,:])
    E_vals = k_vals**2/2
    theta_vals = np.real(PAD[1,:])
    phi_vals = np.real(PAD[2,:])
    pad_vals = np.real(PAD[3,:])

    kx_vals = k_vals*np.sin(theta_vals)*np.cos(phi_vals)
    ky_vals = k_vals*np.sin(theta_vals)*np.sin(phi_vals)


    PAD_dict = {}
    for i,E in enumerate(E_vals):
        PAD_dict[(E,theta_vals[i],phi_vals[i])] = pad_vals[i]
    
    asymmetry_vals = []
    for key,value in PAD_dict.items():
        E,theta,phi = key
        pad_val = value
        opposite_phi = phi_range[np.argmin(np.abs(((phi+np.pi) % (2*np.pi))-phi_range))]

        pad_val_opposite = PAD_dict[(E,theta,opposite_phi)]
        A = (pad_val-pad_val_opposite)/(pad_val+pad_val_opposite)
        asymmetry_vals.append(A)

    asymmetry_vals = np.array(asymmetry_vals)
   
    plt.scatter(kx_vals, ky_vals, c=asymmetry_vals, cmap="bwr", vmin=-1, vmax=1)
    plt.colorbar()
    plt.savefig("images/A.png")

    plt.clf()

    plt.scatter(E_vals,phi_vals,c=asymmetry_vals, cmap="bwr",vmin = -1,vmax = 1)
    plt.xlim([0,0.8])
    plt.colorbar()
    plt.savefig("images/A_rect.png")

    plt.clf()



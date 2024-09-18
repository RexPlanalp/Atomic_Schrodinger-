import numpy as np
import matplotlib.pyplot as plt


import pickle


from scipy.special import sph_harm
from scipy.optimize import least_squares


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
phi_range = np.arange(-2*np.pi, 2 * np.pi, 0.01)  # Range of phi values
E_target = 0.48  # The target energy


# Find the closest energy index to E = 0.48
E_idx = np.argmin(np.abs(E_range - E_target))
k = k_range[E_idx]  # The corresponding momentum value
E = E_range[E_idx]  # The corresponding energy value


# lm values restricted to a specific subrange
lm_vals = [(26,26),(25,25),(24,24),(23,23),(22,22),(21,21)]


# Initialize lists for phi values and asymmetry values


total_amplitudes = {}
total_phases = {}

for key, value in partial_spectra.items():
    l, m = key
    if (l, m) not in lm_vals:
        continue
        
    total_amplitudes[(l, m)] = np.abs((-1j) ** l * np.exp(1j * phases[(E, l)]) * value[E_idx])
    total_phases[(l, m)] = np.angle((-1j) ** l * np.exp(1j * phases[(E, l)]) * value[E_idx])


pad_vals = []


for phi in phi_range:
   
    
    pad_amp = 0
   
    
    # Compute partial wave expansions for forward and opposite directions
    for key, value in partial_spectra.items():
        l, m = key
        if (l, m) not in lm_vals:
            continue
        
       
        pad_amp += total_amplitudes[key] * np.exp(1j*total_phases[key]) * sph_harm(m, l, phi, np.pi / 2) 
         
    
    # Compute the photoelectron angular distributions
    pad_val = np.abs(pad_amp) ** 2

    pad_vals.append(pad_val)
    
pad_vals = np.array(pad_vals)
  




# Plot phi vs asymmetry (Original Data)
plt.plot(phi_range, pad_vals, label=f'PAD at E = {E_target} a.u.', color='b')
plt.xlabel('Phi (radians)')
plt.ylabel('PAD')
plt.title(f'PAD vs Phi at E = {E_target} a.u.')
plt.legend()
plt.savefig("images/PAD_slice.png")
plt.clf()



#############################################################################################################
#############################################################################################################


fixed_lm_vals = []
fitted_lm_vals = [(l,m) for l,m in lm_vals if (l,m) not in fixed_lm_vals]

n_params = 2 * len(fitted_lm_vals)  

print(f"Fitting {n_params} parameters")
print("\n")

# Define the model function for fitting
def pad_model(params, phi_vals, theta=np.pi/2):
    pad_amp = 0
    

    amplitudes = params[:len(fitted_lm_vals)]
    phases = params[len(fitted_lm_vals):]

    

    for i, (l, m) in enumerate(lm_vals):
        if (l,m) in fixed_lm_vals:
            amplitude = total_amplitudes[(l, m)]
            phase = total_phases[(l, m)]
        else:
            idx = fitted_lm_vals.index((l,m))
            amplitude = amplitudes[idx]
            phase = phases[idx]

        sph_h = sph_harm(m, l, phi_vals, theta)
        pad_amp += amplitude * np.exp(1j * phase) * sph_h
       

    pad_val = np.abs(pad_amp) ** 2
    
    

    return pad_val

# Define the least squares residual function
def pad_model_least_squares(params, phi_vals, pad_vals):
    pad_model_vals = pad_model(params, phi_vals)
    return pad_model_vals - pad_vals  

num_fits = 10
ftol = 1e-15
xtol = 1e-15
gtol = 1e-15

# Best result tracker
best_result = None
best_cost = np.inf  

from tqdm import tqdm

import time
start = time.time()
for i in tqdm(range(num_fits)):
    print(f"Working on Fit: {i}")

    # Random initial guesses
    # Random initial guesses
    amplitude_guess = np.zeros(len(fitted_lm_vals))

    for j in range(len(fitted_lm_vals)):
        if j == 1:
            amplitude_guess[j] = np.random.uniform(0, 1)
        elif j > 1 and j!= len(fitted_lm_vals)-1:
            amplitude_guess[j] = np.random.uniform(0, amplitude_guess[j-1])
        else:
            amplitude_guess[j] = np.random.uniform(0, amplitude_guess[-1])
    phase_guess = np.random.uniform(low=0, high=2*np.pi-0.1, size=len(fitted_lm_vals))
    initial_guess = np.concatenate((amplitude_guess, phase_guess))
    
    # Bounds
    n_params = len(initial_guess)
    lower_bounds = [0] * len(fitted_lm_vals) + [0] * len(fitted_lm_vals)
    upper_bounds = [1] * len(fitted_lm_vals) + [2*np.pi-0.1] * len(fitted_lm_vals)
    bounds = (lower_bounds, upper_bounds)
    
    # Least squares fitting
    result = least_squares(pad_model_least_squares, initial_guess,
                           args=(phi_range, pad_vals),
                           max_nfev=500000,
                           bounds=bounds,
                           ftol=ftol, xtol=xtol, gtol=gtol)
    
    # Track the best result
    if result.cost < best_cost:
        best_cost = result.cost
        best_result = result

fitted_amplitudes = best_result.x[:len(fitted_lm_vals)]
fitted_phases = best_result.x[len(fitted_lm_vals):]

end = time.time()
print(f"Time taken: {end-start} seconds")

print(f"Fit Amplitude: {fitted_amplitudes}")
print(f"Exact Amplitude: {[total_amplitudes[(l,m)] for l,m in fitted_lm_vals]}")

print(f"Fit Phase: {fitted_phases}")
print(f"Exact Phase: {[total_phases[(l,m)] for l,m in fitted_lm_vals]}")

# Plot comparison of phases and amplitudes
plt.figure(figsize=(12, 6))

# Subplot 1: Comparison of Phases
plt.subplot(1, 2, 1)
phase_data = plt.scatter([f'l={l}' for l, m in fitted_lm_vals], np.array(fitted_phases)%(2*np.pi), color='b', label='Fitted Phase')
phase_fit = plt.scatter([f'l={l}' for l, m in fitted_lm_vals], np.array([total_phases[(l,m)] for l,m in fitted_lm_vals])%(2*np.pi), color='r', label='Exact Phase', marker='x')
plt.ylim([0, 2*np.pi+0.2])
plt.xlabel('l, m values')
plt.ylabel('Phase (radians)')
plt.title('Comparison of Phases')
plt.legend()

# Subplot 2: Comparison of Amplitudes
plt.subplot(1, 2, 2)
amp_data = plt.scatter([f'l={l}' for l, m in fitted_lm_vals], np.array(fitted_amplitudes), color='b', label='Fitted Amplitude')
amp_fit = plt.scatter([f'l={l}' for l, m in fitted_lm_vals], np.array([total_amplitudes[(l,m)]for l,m in fitted_lm_vals]), color='r', label='Exact Amplitude', marker='x')
plt.xlabel('l, m values')
plt.ylabel('Amplitude')
plt.title('Comparison of Amplitudes')
plt.legend()

plt.tight_layout()
plt.savefig("images/pad_fit_parameters.png")
plt.show()


# Generate fitted asymmetry values using the optimized parameters

pad_fitted = pad_model(result.x, phi_range)

# Plot the original data and the fitted model
plt.figure(figsize=(10, 5))
plt.plot(phi_range, pad_vals, label='Computed Asymmetry', color='blue')
plt.plot(phi_range, pad_fitted, label='Fitted Asymmetry', color='red', linestyle='--')
plt.xlabel('Phi (radians)')
plt.ylabel('Asymmetry (A)')
plt.title(f'Asymmetry vs Phi at E = {E_target} a.u.')
plt.legend()
plt.savefig("images/pad_fit.png")
plt.show()
plt.clf()
############################################
############################################
############################################



import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.special import sph_harm
from scipy.optimize import least_squares
from tqdm import tqdm
import os

if not os.path.exists("fit_images"):
    os.makedirs("fit_images")
# Shooting Method For Coulomb Wave Function
def Shooting_Method_faster(r_range, l, E,pot):
    r_range2 = r_range**2
    dr = r_range[1] - r_range[0]
    dr2 = dr * dr
    
    l_term = l * (l + 1)
    k = np.sqrt(2 * E)
    potential = np.empty_like(r_range)
    potential[0] = np.inf  # Set the potential at r=0 to a high value to avoid division by zero.

    if pot == "H":
        potential[1:] = -1 / r_range[1:]
        z = 1
   

    coul_wave = np.zeros_like(r_range)
    coul_wave[0] = 1.0
    # Adjust initialization for the second point if r_range starts from 0
    coul_wave[1] = coul_wave[0] * (dr2 * (l_term / r_range2[1] + 2 * potential[1] + 2 * E) + 2)

    # Compute wave function values
    for idx in range(2, len(r_range)):
        term = dr2 * (l_term / r_range2[idx-1] + 2 * potential[idx-1] - 2 * E)
        coul_wave[idx] = coul_wave[idx-1] * (term + 2) - coul_wave[idx-2]

    # Final values and phase computation
    r_val = r_range[-2]
    coul_wave_r = coul_wave[-2]
    dcoul_wave_r = (coul_wave[-1] - coul_wave[-3]) / (2 * dr)
    
    norm = np.sqrt(np.abs(coul_wave_r)**2 + (np.abs(dcoul_wave_r) / (k + 1 / (k * r_val)))**2)
    phase = np.angle((1.j * coul_wave_r + dcoul_wave_r / (k + 1 / (k * r_val))) /
                     (2 * k * r_val)**(1.j * 1 / k)) - k * r_val + l * np.pi / 2

    coul_wave /= norm
    return phase, coul_wave

# Finds channel closings for a given energy and laser parameters
def findPhotons(E):
    Ip = -0.5 # Ionization potential of species in atomic units
    N = 10 # Numbe08r of cycles of laser pulse 
    w = 0.057 # Central Frequency of laser pulse in atomic units
    I = 2e14 / 3.51E16 # Intensity of laser pulse in atomic units

    tau = 2*np.pi/w # Period of laser pulse in atomic units
    t = np.linspace(0,tau,1000) # Time array
    envelope = np.sin(np.pi*t/tau)**2 # Envelope function, usually Sin2
    I_profile = I * envelope # Intensity profile of laser pulse
    Up = I_profile/(4*w**2) # Pondermotive energy in atomic units


    n = (E-Ip+Up)/w 
    return n

# Phases and partial spectra for constructing data to fit against
with open('PES_files/partial_spectra.pkl', 'rb') as file:
    partial_spectra = pickle.load(file)
with open('PES_files/phases.pkl', 'rb') as file:
    phases = pickle.load(file)

# Load the full energy range and compute momentum range
E_range = np.load("PES_files/E_range.npy")

# Define parameters for asymmetry slice
theta = np.pi / 2  # Fixed theta = pi/2
phi_range = np.arange(-2*np.pi, 2 * np.pi, 0.01)  # Range of phi values

# Range to expand the Coulomb wave function's on
r_range = np.arange(0, 1500 + 0.01, 0.01)



############################################################################################################################

# Target block of wavefunction to reconstruct
target_lm = (25, 25)  

# Energy range over which channel closings correspond to target_lm
E_fit = []
for E in E_range:
    n = findPhotons(E)
    max_search = int(np.round(np.max(n))+1)
    min_search = int(np.round(np.max(n))-3)

    if target_lm[0] in range(min_search,max_search+1):
        E_fit.append(E)


# Compute the Coulomb wave functions required to reconstruct the target block
cont_states = {}
for E in E_fit:
    print(f"Cont. State for E = {E}")
    phase, coul_wave = Shooting_Method_faster(r_range,target_lm[0], E, "H")
    cont_states[(E,target_lm[0])] = coul_wave

# Populate dict. with expansion coefficients for cont. states
coefficient_dict = {}

for E_target in E_fit:
    print(f"Starting Fitting Process for E = {E_target}")

    # Determine what l,m values to fit against
    n = findPhotons(E_target)
    max_search = int(np.round(np.max(n))+1)
    min_search = int(np.round(np.max(n))-3)
    lm_vals = [(i,i) for i in range(min_search,max_search+1)]
    lm_vals.reverse()

    print(f"lm_vals: {lm_vals}")

    E_idx = np.argmin(np.abs(E_range - E_target))

    # Construct the data to fit against
    A_vals = []

    total_amplitudes = {}
    total_phases = {}

    for key, value in partial_spectra.items():
        l, m = key
        if (l, m) not in lm_vals:
            continue
            
        total_amplitudes[(l, m)] = np.abs((-1j) ** l * np.exp(1j * phases[(E, l)]) * value[E_idx])
        total_phases[(l, m)] = np.angle((-1j) ** l * np.exp(1j * phases[(E, l)]) * value[E_idx])

    # Loop over phi to compute the asymmetry
    for phi in phi_range:
        
        pad_amp = 0
        pad_amp_opp = 0
        
        # Compute partial wave expansions for forward and opposite directions
        for key, value in partial_spectra.items():
            l, m = key
            if (l, m) not in lm_vals:
                continue
            
        
            pad_amp += total_amplitudes[key] * np.exp(1j*total_phases[key]) * sph_harm(m, l, phi, np.pi / 2) 
            pad_amp_opp += total_amplitudes[key] * np.exp(1j*total_phases[key]) * sph_harm(m, l, phi+np.pi, np.pi / 2) 
        
        # Compute the photoelectron angular distributions
        pad_val = np.abs(pad_amp) ** 2
        pad_val_opp = np.abs(pad_amp_opp) ** 2
        
        # Calculate asymmetry A = (PAD - PAD_opp) / (PAD + PAD_opp)
        A = (pad_val - pad_val_opp) / (pad_val + pad_val_opp)
        A_vals.append(A)


    A_vals = np.array(A_vals)
#############################################################################################################
#############################################################################################################


    fixed_lm_vals = []
    fitted_lm_vals = [(l,m) for l,m in lm_vals if (l,m) not in fixed_lm_vals]

    n_params = 2 * len(fitted_lm_vals)  

    # Define the model function for fitting
    def asymmetry_model(params, phi_vals, theta=np.pi/2):
        pad_amp = 0
        pad_amp_opp = 0

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
            sph_h_opp = sph_harm(m, l, phi_vals + np.pi, theta)
            pad_amp_opp += amplitude * np.exp(1j * phase) * sph_h_opp

        pad_val = np.abs(pad_amp) ** 2
        pad_val_opp = np.abs(pad_amp_opp) ** 2
        asymmetry = (pad_val - pad_val_opp) / (pad_val + pad_val_opp)

        return asymmetry

    # Define the least squares residual function
    def asymmetry_model_least_squares(params, phi_vals, A_vals):
        A_model = asymmetry_model(params, phi_vals)
        return A_model - A_vals  

    num_fits = 50

    # Best result tracker
    best_result = None
    best_cost = np.inf  

    for i in tqdm(range(num_fits)):
        print(f"Working on Fit: {i}")

        # Random initial guesses
        # amplitude_guess = np.zeros(len(fitted_lm_vals))

        # for j in range(len(fitted_lm_vals)):
        #     if j == 1:
        #         amplitude_guess[j] = np.random.uniform(0, 1)
        #     elif j > 1 and j!= len(fitted_lm_vals)-1:
        #         amplitude_guess[j] = np.random.uniform(0, amplitude_guess[j-1])
        #     else:
        #         amplitude_guess[j] = np.random.uniform(0, amplitude_guess[-1])

        
        amplitude_guess = np.zeros(len(fitted_lm_vals))

        for j in range(len(fitted_lm_vals)):
            if j == 1:
                amplitude_guess[j] = 1  # Set the value at idx to 1
            else:
                amplitude_guess[j] = np.random.uniform(0, 1-0.15)  # Random value between 0 and 1

        #amplitude_guess = np.random.uniform(low=0, high=1, size=len(fitted_lm_vals))
        phase_guess = np.random.uniform(low=0, high=2*np.pi-0.1, size=len(fitted_lm_vals))
        initial_guess = np.concatenate((amplitude_guess, phase_guess))
        
        # Bounds
        n_params = len(initial_guess)
        lower_bounds = [0] * len(fitted_lm_vals) + [0] * len(fitted_lm_vals)
        upper_bounds = [1] * len(fitted_lm_vals) + [2*np.pi-0.1] * len(fitted_lm_vals)
        bounds = (lower_bounds, upper_bounds)
        
        # Least squares fitting
        result = least_squares(asymmetry_model_least_squares, initial_guess,
                            args=(phi_range, A_vals),
                            max_nfev=500000,
                            bounds=bounds,
                            ftol=1e-15, xtol=1e-15, gtol=1e-15)
        
        # Track the best result
        if result.cost < best_cost:
            best_cost = result.cost
            best_result = result

    fitted_amplitudes = best_result.x[:len(fitted_lm_vals)]
    fitted_phases = best_result.x[len(fitted_lm_vals):]

    target_amplitude = fitted_amplitudes[lm_vals.index(target_lm)]
    target_phase = fitted_phases[lm_vals.index(target_lm)]

    target_phase *= np.conjugate(np.angle((-1j) ** (target_lm[0]) * np.exp(1j * phases[(E_target, (target_lm[0]))])))

    coefficient_dict[E_target] = (target_amplitude, target_phase)

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
    plt.savefig(f"fit_images/{E_target}_fit.png")
    plt.clf()

wavefunction = np.zeros_like(r_range, dtype=np.complex128)
for E in E_fit:
    amplitude, phase = coefficient_dict[E]
    wavefunction += amplitude * cont_states[(E, target_lm[0])] * np.exp(1j * phase)

plt.plot(r_range, np.abs(wavefunction)**2)
plt.savefig("test_reconstruction.png")

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.special import sph_harm
from scipy.optimize import least_squares

# Load partial spectra and phases from simulation data
with open('PES_files/partial_spectra.pkl', 'rb') as file:
    partial_spectra = pickle.load(file)
with open('PES_files/phases.pkl', 'rb') as file:
    phases = pickle.load(file)

# Finds channel closings for a given energy and laser parameters
def findPhotons(E):
    Ip = -0.5  # Ionization potential of species in atomic units
    N = 10  # Number of cycles of laser pulse
    w = 0.057  # Central Frequency of laser pulse in atomic units
    I = 2e14 / 3.51E16  # Intensity of laser pulse in atomic units

    tau = 2 * np.pi / w  # Period of laser pulse in atomic units
    t = np.linspace(0, tau, 1000)  # Time array
    envelope = np.sin(np.pi * t / tau) ** 2  # Envelope function, usually Sin^2
    I_profile = I * envelope  # Intensity profile of laser pulse
    Up = I_profile / (4 * w ** 2)  # Ponderomotive energy in atomic units

    n = (E - Ip + Up) / w
    return n

# Load energy range and compute momentum range
E_range = np.load("PES_files/E_range.npy")
k_range = np.sqrt(2 * E_range)

# Define parameters, depends on polarization
theta = np.pi / 2  # Fixed theta = pi/2
phi_range = np.arange(0, 2 * np.pi, 0.01)  # Range of phi values

# Energy to compute and fit asymmetry at
E_target = 0.48  # The target energy

# Find the closest energy index to E_target
E_idx = np.argmin(np.abs(E_range - E_target))
k = k_range[E_idx]  # The corresponding momentum value
E = E_range[E_idx]  # The corresponding energy value

# Use findPhotons to compute channel closings and predict l,m values to use for fit
n = findPhotons(E)
max_search = int(np.round(np.max(n)) + 1)
min_search = int(np.round(np.max(n)) - 4)
lm_vals = [(i, i) for i in range(min_search, max_search + 1)]
lm_vals.reverse()
print(f"Fitting for lm_vals = {lm_vals}")

# Initialize lists for asymmetry values from simulation
A_vals = []

# Define new dictionaries to hold the amplitudes and phases for E_target specifically
total_amplitudes = {}
total_phases = {}

for key, value in partial_spectra.items():
    l, m = key
    if (l, m) not in lm_vals:
        continue

    total_amplitudes[(l, m)] = np.abs((-1j) ** l * np.exp(1j * phases[(E, l)]) * value[E_idx])
    total_phases[(l, m)] = np.angle((-1j) ** l * np.exp(1j * phases[(E, l)]) * value[E_idx])

# Loop over phi to compute the asymmetry based on the amplitudes and phases at E_target
for phi in phi_range:
    pad_amp = 0
    pad_amp_opp = 0

    # Compute partial wave expansions for forward and opposite directions
    for (l, m) in lm_vals:
        amplitude = total_amplitudes[(l, m)]
        phase = total_phases[(l, m)]
        pad_amp += amplitude * np.exp(1j * phase) * sph_harm(m, l, phi, np.pi / 2)
        pad_amp_opp += amplitude * np.exp(1j * phase) * sph_harm(m, l, phi + np.pi, np.pi / 2)

    # Compute the photoelectron angular distributions
    pad_val = np.abs(pad_amp) ** 2
    pad_val_opp = np.abs(pad_amp_opp) ** 2

    # Calculate asymmetry A = (PAD - PAD_opp) / (PAD + PAD_opp)
    A = (pad_val - pad_val_opp) / (pad_val + pad_val_opp)
    A_vals.append(A)

# Convert the lists to numpy arrays
A_vals = np.array(A_vals)

# Plot phi vs asymmetry (Original Data)
plt.plot(phi_range, A_vals, label=f'Asymmetry at E = {E} a.u.', color='b')
plt.xlabel('Phi (radians)')
plt.ylabel('Asymmetry (A)')
plt.title(f'Asymmetry vs Phi at E = {E} a.u.')
plt.legend()
plt.savefig("images/A_slice.png")
plt.clf()

# Done with data generation, now we will fit the data
#############################################################################################################
#############################################################################################################

# Define the list of fixed lm values and fitted lm values
fixed_lm_vals = []
fitted_lm_vals = [(l, m) for l, m in lm_vals if (l, m) not in fixed_lm_vals]

# Define the number of parameters to fit (only amplitudes now)
n_params = len(fitted_lm_vals)

print(f"Fitting {n_params} parameters")
print("\n")

# Define the model function for fitting
def asymmetry_model(params, phi_vals, theta=np.pi / 2):
    pad_amp = 0
    pad_amp_opp = 0

    amplitudes = params  # Only amplitudes are fitted

    for i, (l, m) in enumerate(lm_vals):
        if (l, m) in fixed_lm_vals:
            amplitude = total_amplitudes[(l, m)]
        else:
            idx = fitted_lm_vals.index((l, m))
            amplitude = amplitudes[idx]

        phase = total_phases[(l, m)]  # Phases are fixed and read from the dictionary

        sph_h = sph_harm(m, l, phi_vals, theta)
        pad_amp += amplitude * np.exp(1j * phase) * sph_h
        sph_h_opp = sph_harm(m, l, phi_vals + np.pi, theta)
        pad_amp_opp += amplitude * np.exp(1j * phase) * sph_h_opp

    pad_val = np.abs(pad_amp) ** 2
    pad_val_opp = np.abs(pad_amp_opp) ** 2
    asymmetry = (pad_val - pad_val_opp) / (pad_val + pad_val_opp + 1e-10)

    return asymmetry

# Define the least squares residual function
def asymmetry_model_least_squares(params, phi_vals, A_vals):
    A_model = asymmetry_model(params, phi_vals)
    return A_model - A_vals

# Define the number of fits to perform
num_fits = 150

# Best result tracker
best_result = None
best_cost = np.inf

for i in range(num_fits):
    print(f"Working on Fit: {i + 1}")

    # Make a random guess of amplitude values
    # Amplitudes are between 0 and 0.7
    amplitude_guess = np.random.uniform(low=0, high=0.7, size=len(fitted_lm_vals))
    initial_guess = amplitude_guess

    # Set bounds on the amplitude values
    lower_bounds = [0] * len(fitted_lm_vals)
    upper_bounds = [0.7] * len(fitted_lm_vals)
    bounds = (lower_bounds, upper_bounds)

    # Run the fitting algorithm
    result = least_squares(
        asymmetry_model_least_squares,
        initial_guess,
        args=(phi_range, A_vals),
        max_nfev=500000,
        bounds=bounds,
        ftol=1e-15,
        xtol=1e-15,
        gtol=1e-15,
    )

    # Track the best result
    if result.cost < best_cost:
        best_cost = result.cost
        best_result = result

# Extract the fitted amplitudes for the best fit
fitted_amplitudes = best_result.x

# Plot comparison of amplitudes
plt.figure(figsize=(8, 6))
plt.title(f'Fitted Amplitudes vs Exact Amplitudes at E = {E} a.u.')

# Comparison of Amplitudes
plt.scatter(
    [f'l={l}' for l, m in fitted_lm_vals],
    np.array(fitted_amplitudes)/np.array(np.max(fitted_amplitudes)),
    color='b',
    label='Fitted Ratios',
)
plt.scatter(
    [f'l={l}' for l, m in fitted_lm_vals],
    np.array([total_amplitudes[(l, m)] for l, m in fitted_lm_vals])/np.max([total_amplitudes[(l, m)] for l, m in fitted_lm_vals]),
    color='r',
    label='Exact Ratios',
    marker='x',
)
plt.xlabel('l, m values')
plt.ylabel('Ratio')
plt.title(f'Comparison of Ratios')
plt.legend()

plt.tight_layout()
plt.savefig(f"images/fit_parameters_{E}.png")
plt.show()

# Generate fitted asymmetry values using the optimized parameters
A_fitted = asymmetry_model(best_result.x, phi_range)

# Plot the original data and the fitted model
plt.figure(figsize=(10, 5))
plt.plot(phi_range, A_vals, label='Computed Asymmetry', color='blue')
plt.plot(phi_range, A_fitted, label='Fitted Asymmetry', color='red', linestyle='--')
plt.xlabel('Phi (radians)')
plt.ylabel('Asymmetry (A)')
plt.title(f'Asymmetry vs Phi at E = {E} a.u.')
plt.legend()
plt.savefig(f"images/A_fit_{E}.png")
plt.show()
plt.clf()

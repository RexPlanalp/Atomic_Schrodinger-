import numpy as np
import matplotlib.pyplot as plt


import sys
import pickle




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


E_target = float(sys.argv[1])  # The target energy

# Find the closest energy index to E = 0.48
E_idx = np.argmin(np.abs(E_range - E_target))
k = k_range[E_idx]  # The corresponding momentum value
E = E_range[E_idx]  # The corresponding energy value




# Initialize lists for phi values and asymmetry values

lm_vals = []
amplitudes = []


for key, value in partial_spectra.items():
    l, m = key
    condition = l ==m
    if not condition:
        continue
    amplitudes.append( np.abs((-1j) ** l * np.exp(1j * phases[(E, l)]) * value[E_idx]))
    lm_vals.append((l, m))

sorted_indices = np.argsort(amplitudes)[::-1]

sorted_amplitudes = np.array(amplitudes)[sorted_indices]
sorted_lm_vals = np.array(lm_vals)[sorted_indices]

sorted_l_vals = np.array([l for l, m in sorted_lm_vals])

mask = (sorted_l_vals >= 22) & (sorted_l_vals <= 30)


# Scatter plot
plt.scatter(sorted_l_vals[mask], sorted_amplitudes[mask], s=2, color="black")
plt.ylabel("Partial Wave Amplitude")
plt.ylim([0, np.max(sorted_amplitudes[mask]) + 0.1])
plt.xticks([])
plt.title(f"(l,m) Partial Wave Amplitudes at E = {E_target}")


# Adding small text labels for each point with (l, m)
for i, (l, m) in enumerate(sorted_lm_vals[mask]):
    plt.text(l, sorted_amplitudes[i]+0.01, f"({l},{m})", fontsize=10, ha='center')


# Add legend and save the figure
plt.legend([f"Max: {sorted_lm_vals[np.argmax(sorted_amplitudes)]}"], loc="upper right")
plt.savefig("images/top.png")





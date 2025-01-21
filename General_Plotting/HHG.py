import numpy as np
import matplotlib.pyplot as plt
import json

# Path to your JSON file
file_path = "input.json"

# Open and load the JSON file
with open(file_path, 'r') as file:
    input_data = json.load(file)

polarization = input_data["laser"]["polarization"]

I_max = input_data["laser"]["I"]/3.51E16
w = input_data["laser"]["w"]
N = input_data["box"]["N"]

if input_data["species"] == "H":
    Ip = -0.5
elif input_data["species"] == "Ar":
    Ip = -0.5791546178

Up = I_max/(4*w**2)

cut_off = 3.17*Up - Ip


# Load data
HHG_data = np.real(np.load("TDSE_files/HHG_data.npy"))      # Shape: (3, N)
laser_data = np.real(np.load("TDSE_files/laser_data.npy"))  # Shape: (3, N)
total_time = np.real(np.load("TDSE_files/time.npy"))     # Shape: (N,)

# Initialize dipole_acceleration as a 2D array to store each component
dipole_acceleration = np.zeros((3, len(total_time)))

# Calculate dipole acceleration for each component
for i in range(3):
    dipole_acceleration[i, :] = -HHG_data[i, :] + np.gradient(laser_data[i, :], total_time)

# Plot the dipole acceleration components (optional)
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(total_time, dipole_acceleration[i, :], label=f'Component {i+1}')
plt.xlabel('Time (a.u.)')
plt.ylabel('Dipole Acceleration (a.u.)')
plt.title('Dipole Acceleration Components')
plt.legend()
plt.savefig("images/dipole_accel.png")
plt.clf()

# Apply a window function (Blackman window) to each component
window = np.blackman(len(total_time))
windowed_dipole = dipole_acceleration * window  # Broadcasting applies the window to each row

# Perform FFT on each component
dipole_fft = np.fft.fft(windowed_dipole, axis=1)
frequencies = np.fft.fftfreq(len(total_time), d=total_time[1] - total_time[0])

# Compute the power spectrum (magnitude squared of FFT) for each component
power_spectrum = np.abs(dipole_fft)**2

# Sum the power spectra of all components to get the total power spectrum
total_power_spectrum = np.sum(power_spectrum, axis=0)

# Only keep the positive frequencies
positive_freq_idx = frequencies > 0
frequencies = frequencies[positive_freq_idx]
total_power_spectrum = total_power_spectrum[positive_freq_idx]

# Plot the total harmonic spectrum
plt.figure(figsize=(8, 6))
plt.semilogy(frequencies * 2*np.pi / w, total_power_spectrum, color='b')
plt.axvline(cut_off/w, color='r', linestyle='--', label='Cut-off Energy')
plt.xlim([0, 60])        # Adjust based on your data's frequency range
plt.ylim([1e-4, 1e4])     # Adjust based on the power spectrum's range
plt.xlabel('Frequency (atomic units)')
plt.ylabel('Intensity (arb. units)')
plt.title('Harmonic Spectrum')
plt.grid(True, which='both', ls='--', lw=0.5)
plt.savefig("images/harmonic_spectrum.png")
plt.clf()

print("Dipole acceleration components and harmonic spectrum have been successfully processed and saved.")


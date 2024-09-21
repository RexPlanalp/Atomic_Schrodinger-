import numpy as np
import matplotlib.pyplot as plt

# Load data
HHG_data = np.real(np.load("TDSE_files/HHG_data.npy"))
laser_data = np.real(np.load("TDSE_files/laser_data.npy"))
total_time = np.real(np.load("TDSE_files/t_total.npy"))


# Calculate dipole acceleration
dipole_acceleration = -HHG_data[2,:] + np.gradient(laser_data[2,:], total_time)

# Plot the harmonic spectrum
plt.figure(figsize=(8, 6))
plt.plot(total_time,dipole_acceleration)
plt.savefig("images/dipole_accel.png")
plt.clf()

# Apply a window function (Blackman window to reduce spectral leakage)
window = np.blackman(len(dipole_acceleration))
windowed_dipole = dipole_acceleration * window

# Perform FFT
dipole_fft = np.fft.fft(windowed_dipole)
frequencies = np.fft.fftfreq(len(dipole_acceleration), d=total_time[1] - total_time[0])

# Compute the power spectrum (magnitude squared of FFT)
power_spectrum = np.abs(dipole_fft)**2

# Only keep the positive frequencies
positive_freq_idx = np.where(frequencies > 0)
frequencies = frequencies[positive_freq_idx]
power_spectrum = power_spectrum[positive_freq_idx]

# Apply a window function (Blackman window to reduce spectral leakage)
window = np.blackman(len(dipole_acceleration))
windowed_dipole = dipole_acceleration * window

# Perform FFT
dipole_fft = np.fft.fft(windowed_dipole)
frequencies = np.fft.fftfreq(len(dipole_acceleration), d=total_time[1] - total_time[0])

# Compute the power spectrum (magnitude squared of FFT)
power_spectrum = np.abs(dipole_fft)**2

# Only keep the positive frequencies
positive_freq_idx = np.where(frequencies > 0)
frequencies = frequencies[positive_freq_idx]
power_spectrum = power_spectrum[positive_freq_idx]

# Plot the harmonic spectrum
plt.figure(figsize=(8, 6))
plt.semilogy(frequencies, power_spectrum)
plt.xlim([0,0.6])
plt.ylim([1e-4,1e4])
plt.xlabel('Frequency (atomic units)')
plt.ylabel('Intensity (arb. units)')
plt.title('Harmonic Spectrum')
plt.savefig("images/harmonic_spectrum.png")


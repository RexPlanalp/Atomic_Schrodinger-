import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import find_peaks
import json
import sys


file_path = "input.json"

# Open and load the JSON file
with open(file_path, 'r') as file:
    input_data = json.load(file)

w = input_data["lasers"]["w"]
dE = input_data["E"][0]

PEAKS = "PEAKS" in sys.argv

theta = np.pi / 2  
phi_range = np.arange(0, 2 * np.pi, 0.01)  


def findPeakIndices(PES,dE,w):
    d = int(w/dE)
    peak_indices = find_peaks(PES,distance = d)
    return peak_indices[0]

E_range = np.load("PES_files/E_range.npy")
PES = np.real(np.load("PES_files/PES.npy"))
PAD = np.real(np.load("PES_files/PAD.npy"))

peak_indices = findPeakIndices(PES,dE,w)

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

    # Relative Asymmetry
    A = (pad_val-pad_val_opposite)/(pad_val+pad_val_opposite)

    # Logarithmic Asymmetry
    #A = np.log(pad_val/pad_val_opposite)

    # Absolute Asymmetry 
    #A = pad_val-pad_val_opposite

    # SMAPE 
    #A = np.abs(pad_val-pad_val_opposite)/((np.abs(pad_val)+np.abs(pad_val_opposite))/2)

    # Scaled RMSE

    #A = np.sqrt((pad_val-pad_val_opposite)**2)/np.sqrt((pad_val**2+pad_val_opposite**2)/2)


    asymmetry_vals.append(A)

asymmetry_vals = np.array(asymmetry_vals)


asymmetry_slice = []
temp_vals = []
E_input = float(sys.argv[1])
E_target = E_range[np.argmin(np.abs(E_range - E_input))]

for idx, E_val in enumerate(E_vals):
    if np.isclose(E_val, E_target):
        temp_vals.append(phi_vals[idx])
        asymmetry_slice.append(asymmetry_vals[idx])

# Convert lists to numpy arrays for sorting
temp_vals = np.array(temp_vals)
asymmetry_slice = np.array(asymmetry_slice)

# Sort the asymmetry_slice based on temp_vals (phi values)
sorted_indices = np.argsort(temp_vals)
temp_vals = temp_vals[sorted_indices]
asymmetry_slice = asymmetry_slice[sorted_indices]




plt.scatter(E_vals,phi_vals,c=asymmetry_vals, cmap="bwr")
if PEAKS:
    for peak_idx in peak_indices:
        plt.axvline(x=E_range[peak_idx],color='black')
plt.ylabel("Phi (radians)")
plt.xlabel("Energy (a.u.)")
plt.xlim([0,1.4])
plt.colorbar()
plt.savefig("images/A_rect.png")
plt.clf()

plt.scatter(kx_vals, ky_vals, c=asymmetry_vals, cmap="bwr")
if PEAKS:
    for peak_idx in peak_indices:
        E_peak = E_range[peak_idx]
        p_peak = np.sqrt(2*E_peak)
        circle = plt.Circle((0, 0), p_peak, color='black', fill=False)
        plt.gca().add_artist(circle)
plt.colorbar()
plt.savefig("images/A_polar.png")
plt.clf()

plt.plot(phi_range,asymmetry_slice)
plt.ylabel(f"Asymmetry at {E_target} ")
plt.xlabel("Phi (radians)")
plt.title(f"Asymmetry at {E_target}")
plt.savefig("images/A_slice.png")
plt.clf()

plt.semilogy(E_range,PES)
for peak_idx in peak_indices:
    if PES[peak_idx] == np.max(PES[peak_indices]):
        color = "black"
    else:
        color = "red"
    plt.plot(E_range[peak_idx],PES[peak_idx],linestyle='none',marker='o',color=color,label = f"{E_range[peak_idx]:.3f}")
plt.legend(fontsize='xx-small')
plt.ylabel("PES")
plt.xlabel("Energy (a.u.)")
plt.savefig("images/PES_peaks.png")
plt.clf()





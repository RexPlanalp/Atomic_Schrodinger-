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
SLICE = input_data["SLICE"]

PEAKS = "PEAKS" in sys.argv

if SLICE == "XZ":
    theta_range = np.arange(0, np.pi, 0.01)
    phi_range = np.array([0, np.pi])
elif SLICE == "XY":
    theta_range = np.array([np.pi / 2])
    phi_range = np.arange(0, 2 * np.pi, 0.01)


E_range = np.load("PES_files/E_range.npy")
PES = np.real(np.load("PES_files/PES.npy"))
PAD = np.real(np.load("PES_files/PAD.npy"))



k_vals = np.real(PAD[0, :])
E_vals = k_vals**2 / 2
theta_vals = np.real(PAD[1, :])
phi_vals = np.real(PAD[2, :])
pad_vals = np.real(PAD[3, :])

kx_vals = k_vals * np.sin(theta_vals) * np.cos(phi_vals)
ky_vals = k_vals * np.sin(theta_vals) * np.sin(phi_vals)
kz_vals = k_vals * np.cos(theta_vals)

# Create PAD_dict with rounded keys to avoid floating point issues
PAD_dict = {}
decimal_places = 5  # Adjust as needed
for i, E in enumerate(E_vals):
    key = (round(E, decimal_places), round(theta_vals[i], decimal_places), round(phi_vals[i], decimal_places))
    PAD_dict[key] = pad_vals[i]

asymmetry_vals = []
for key, pad_val in PAD_dict.items():
    E, theta, phi = key

    # Compute opposite angles
    opposite_phi_val = (phi + np.pi) % (2 * np.pi)
    opposite_theta_val = np.pi - theta

    # Find closest opposite angles
    if SLICE == "XZ":
        opposite_theta = theta_range[np.argmin(np.abs(theta_range - opposite_theta_val))]
        opposite_phi = phi  # Same phi for XZ
    elif SLICE == "XY":
        opposite_phi = phi_range[np.argmin(np.abs(phi_range - opposite_phi_val))]
        opposite_theta = theta  # Same theta for XY

    # Retrieve pad_val_opposite with rounding
    opposite_key = (
        round(E, decimal_places),
        round(opposite_theta, decimal_places),
        round(opposite_phi, decimal_places)
    )
    pad_val_opposite = PAD_dict.get(opposite_key, 0)  # Handle missing keys as needed

    # Compute Asymmetry
    if pad_val + pad_val_opposite != 0:
        A = (pad_val - pad_val_opposite) / (pad_val + pad_val_opposite)
    else:
        A = 0  # or handle division by zero as needed

    asymmetry_vals.append(A)

asymmetry_vals = np.array(asymmetry_vals)

# Slice for specific E
asymmetry_slice = []
temp_vals = []
E_input = float(sys.argv[1])
E_target = E_range[np.argmin(np.abs(E_range - E_input))]

for idx, E_val in enumerate(E_vals):
    if np.isclose(E_val, E_target, atol=1e-5):
        temp_vals.append(phi_vals[idx])
        asymmetry_slice.append(asymmetry_vals[idx])

# Convert lists to numpy arrays for sorting
temp_vals = np.array(temp_vals)
asymmetry_slice = np.array(asymmetry_slice)

# Sort the asymmetry_slice based on temp_vals (phi values)
sorted_indices = np.argsort(temp_vals)
temp_vals = temp_vals[sorted_indices]
asymmetry_slice = asymmetry_slice[sorted_indices]

# Visualization
if SLICE == "XY":
    plt.scatter(kx_vals, ky_vals, c=asymmetry_vals, cmap="bwr", norm=mcolors.CenteredNorm())
elif SLICE == "XZ":
    plt.scatter(kz_vals, kx_vals, c=asymmetry_vals, cmap="bwr", norm=mcolors.CenteredNorm())
plt.colorbar(label='Asymmetry')
plt.xlabel('k_x' if SLICE == "XZ" else 'k_x')
plt.ylabel('k_z' if SLICE == "XZ" else 'k_y')
plt.title(f'Asymmetry in {SLICE} Plane')
plt.savefig("images/A_polar.png")
plt.clf()


plt.scatter(E_vals, phi_vals, c=asymmetry_vals, cmap="bwr")
plt.ylabel("Phi (radians)")
plt.xlabel("Energy (a.u.)")
plt.colorbar(label='Asymmetry')
plt.savefig("images/A_rect.png")
plt.clf()

# plt.plot(phi_range, asymmetry_slice)
# plt.ylabel(f"Asymmetry at {E_target} a.u.")
# plt.xlabel("Phi (radians)")
# plt.title(f"Asymmetry at {E_target} a.u.")
# plt.savefig("images/A_slice.png")
# plt.clf()

# plt.semilogy(E_range, PES)
# for peak_idx in peak_indices:
#     color = "black" if PES[peak_idx] == np.max(PES[peak_indices]) else "red"
#     plt.plot(E_range[peak_idx], PES[peak_idx], linestyle='none', marker='o', color=color, label=f"{E_range[peak_idx]:.3f}")
# plt.legend(fontsize='xx-small')
# plt.ylabel("PES")
# plt.xlabel("Energy (a.u.)")
# plt.savefig("images/peaks.png")
# plt.clf()

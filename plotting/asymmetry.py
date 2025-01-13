import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
import json
import sys

file_path = "input.json"

with open(file_path, 'r') as file:
    input_data = json.load(file)

if input_data["SLICE"] == "XZ":
    theta_range = np.arange(0, np.pi, 0.01)
    phi_range = np.array([0, np.pi])
elif input_data["SLICE"] == "XY":
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
    if input_data["SLICE"] == "XZ":
        opposite_theta = theta_range[np.argmin(np.abs(theta_range - opposite_theta_val))]
        opposite_phi = np.pi-phi
    elif input_data["SLICE"] == "XY":
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


fig,ax = plt.subplots(figsize=(10, 8))
if input_data["SLICE"] == "XY":
    sc = ax.scatter(kx_vals, ky_vals, c=asymmetry_vals, cmap="bwr", norm=CenteredNorm())
elif input_data["SLICE"] == "XZ":
    sc = ax.scatter(kx_vals, kz_vals, c=asymmetry_vals, cmap="bwr", norm=CenteredNorm())
fig.colorbar(sc,ax=ax,label='Asymmetry')
ax.set_xlabel('k_x' if input_data["SLICE"] == "XZ" else 'k_x')
ax.set_ylabel('k_z' if input_data["SLICE"] == "XZ" else 'k_y')
ax.set_title(f'Asymmetry in {input_data["SLICE"]} Plane')
fig.savefig("images/A_polar.png")
fig.clf()


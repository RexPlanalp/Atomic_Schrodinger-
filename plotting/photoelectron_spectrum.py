import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json as json
import sys as sys

from scipy.signal import find_peaks

file_path = "input.json"

# Open and load the JSON file
with open(file_path, 'r') as file:
    input_data = json.load(file)

SLICE = input_data["SLICE"]
w = input_data["laser"]["w"]
dE = input_data["E"][0]
I_au = input_data["laser"]["I"] / 3.51E16
Up_max = I_au/(4*w**2)

LOG_PES = "LOG" == sys.argv[1]
LOG_PAD = "LOG" == sys.argv[2]
CUTOFF = "CUTOFF" == sys.argv[3]

E_range = np.load("PES_files/E_range.npy")
PES = np.real(np.load("PES_files/PES.npy"))
PAD = np.real(np.load("PES_files/PAD.npy"))


if LOG_PES:
    fig,ax = plt.subplots()
    ax.semilogy(E_range,PES,color = "k")
    ax.set_xlabel("Energy (au)")
    ax.set_ylabel("Yield (Log scale)")
    if CUTOFF:
        ax.axvline(2*Up_max,color = "blue",label = "Direct Cutoff")
        ax.axvline(10*Up_max,color = "red",label = "Rescattering Cutoff")
    ax.set_xlim([np.min(E_range),np.max(E_range)])
    ax.legend()
    fig.savefig("images/log_PES.png")
else:
    fig,ax = plt.subplots()
    ax.plot(E_range,PES,color = "k")
    ax.set_xlabel("Energy (au)")
    ax.set_ylabel("Yield (Log scale)")
    if CUTOFF:
        ax.axvline(2*Up_max,color = "blue",label = "Direct Cutoff")
        ax.axvline(10*Up_max,color = "red",label = "Rescattering Cutoff")
    ax.set_xlim([np.min(E_range),np.max(E_range)])
    ax.legend()
    fig.savefig("images/log_PES.png")

k_vals = np.real(PAD[0,:])
theta_vals = np.real(PAD[1,:])
phi_vals = np.real(PAD[2,:])
pad_vals = np.real(PAD[3,:])

kx_vals = k_vals* np.sin(theta_vals) * np.cos(phi_vals)
ky_vals = k_vals * np.sin(theta_vals) * np.sin(phi_vals)
kz_vals = k_vals * np.cos(theta_vals)

max_val = np.max(np.real(pad_vals))
min_val = np.max(np.real(pad_vals))*10**-6

cmap = "hot_r"

if SLICE == "XY":
    fig, ax = plt.subplots()  
    norm = mcolors.LogNorm(vmin=min_val, vmax=max_val) if LOG_PAD else mcolors.Normalize(vmin=min_val, vmax=max_val)
    name = "log_PAD.png" if LOG_PAD else "PAD.png"

    sc = ax.scatter(kx_vals, ky_vals, c=pad_vals, cmap=cmap, norm=norm)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    fig.colorbar(sc, ax=ax)
    fig.savefig("images/" + name)
elif SLICE == "XZ":
    fig, ax = plt.subplots()  
    norm = mcolors.LogNorm(vmin=min_val, vmax=max_val) if LOG_PAD else mcolors.Normalize(vmin=min_val, vmax=max_val)
    name = "log_PAD.png" if LOG_PAD else "PAD.png"

    sc = ax.scatter(kx_vals, kz_vals, c=pad_vals, cmap=cmap, norm=norm)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("kx")
    ax.set_ylabel("kz")
    fig.colorbar(sc, ax=ax)
    fig.savefig("images/" + name)

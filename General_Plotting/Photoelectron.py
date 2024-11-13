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

SLICE = input_data["SLICE"]

LOG_PES = "LOG" == sys.argv[1]
LOG_PAD = "LOG" == sys.argv[2]

PEAKS  = "PEAKS" in sys.argv
w = input_data["lasers"]["w"]
dE = input_data["E"][0]

def findPeakIndices(PES,dE,w):
    d = int(w/dE)
    peak_indices = find_peaks(PES,distance = d)
    return peak_indices[0]

E_range = np.load("PES_files/E_range.npy")
PES = np.real(np.load("PES_files/PES.npy"))
PAD = np.real(np.load("PES_files/PAD.npy"))

peak_indices = findPeakIndices(PES,dE,w)

E_range = np.load("PES_files/E_range.npy")
PES = np.real(np.load("PES_files/PES.npy"))
PAD = np.real(np.load("PES_files/PAD.npy"))


if LOG_PES:
    plt.figure()
    plt.semilogy(E_range,PES,color = "k")
    plt.xlabel("Energy (au)")
    plt.ylabel("Yield (Log scale)")
    plt.savefig("images/log_PES.png")
    plt.clf()
else:
    plt.figure()
    plt.plot(E_range,PES,color = "k")
    plt.xlabel("Energy (au)")
    plt.ylabel("Yield")
    plt.savefig("images/PES.png")
    plt.clf()

if PEAKS:
    peak_indices = find_peaks(PES,width = 2)[0]

    plt.semilogy(E_range,PES)
    for peak_idx in peak_indices:
        if PES[peak_idx] == np.max(PES[peak_indices]):
            color = "black"
        else:
            color = "red"
        plt.plot(E_range[peak_idx],PES[peak_idx],linestyle='none',marker='o',color=color,label = f"{E_range[peak_idx]:.3f}")
    plt.legend(fontsize='x-small')
    plt.ylabel("PES")
    plt.xlabel("Energy (a.u.)")
    plt.savefig("images/PES_peaks.png")
    plt.clf()



PAD = np.load("PES_files/PAD.npy")

k_vals = np.real(PAD[0,:])
theta_vals = np.real(PAD[1,:])
phi_vals = np.real(PAD[2,:])
pad_vals = np.real(PAD[3,:])


kx_vals = k_vals* np.sin(theta_vals) * np.cos(phi_vals)
ky_vals = k_vals * np.sin(theta_vals) * np.sin(phi_vals)
kz_vals = k_vals * np.cos(theta_vals)

max_val = np.max(np.real(pad_vals))
min_val = np.max(np.real(pad_vals))*10**-2

cmap = "plasma"

if SLICE == "XY":
    fig, ax = plt.subplots()  # Create a figure and axes
    cmap_gradient = plt.get_cmap(cmap)  # Get the chosen colormap

    # Set normalization based on LOG_PAD and actual min/max values
    norm = mcolors.LogNorm(vmin=min_val, vmax=max_val) if LOG_PAD else mcolors.Normalize(vmin=min_val, vmax=max_val)

    # Get the smallest color in the colormap actually used by the data
    smallest_color = cmap_gradient(norm(min_val))
    ax.set_facecolor(smallest_color)

    # Create the scatter plot with appropriate normalization
    sc = ax.scatter(kx_vals, ky_vals, c=pad_vals, cmap=cmap, norm=norm)
    
    # Additional plot settings
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    fig.colorbar(sc, ax=ax)

    plt.savefig("images/PAD.png")
elif SLICE == "XZ":
    fig, ax = plt.subplots()  # Create a figure and axes
    cmap_gradient = plt.get_cmap(cmap)  # Get the viridis colormap
    smallest_color = cmap_gradient(0) 


    ax.set_facecolor(smallest_color)
    if LOG_PAD:
        sc = ax.scatter(kz_vals, kx_vals, c=pad_vals, cmap=cmap,norm=mcolors.LogNorm(vmin=min_val,vmax=max_val))
    else:
        sc = ax.scatter(kz_vals, kx_vals, c=pad_vals, cmap=cmap)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("kz")
    ax.set_ylabel("kx")
    fig.colorbar(sc, ax=ax)
    plt.savefig("images/PAD.png")




total_ionization = np.trapz(PES,E_range)
print(total_ionization)

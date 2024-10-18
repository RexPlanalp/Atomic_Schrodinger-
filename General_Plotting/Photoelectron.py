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

LOG = "LOG" in sys.argv
PEAKS  = "PEAKS" in sys.argv



def findPeakIndices(PES):
    peak_indices = find_peaks(PES,width = 2)[0]
    return peak_indices

def findCentralPeakIndex(PES):
    peak_indices = findPeakIndices(PES)
    max_index = np.argmax(PES[peak_indices])
    return max_index

E_range = np.load("PES_files/E_range.npy")
PES = np.real(np.load("PES_files/PES.npy"))
PAD = np.real(np.load("PES_files/PAD.npy"))


if LOG:
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
    max_index = findCentralPeakIndex(PES)

    max_E = E_range[peak_indices][max_index]
    ati_peak_energies = E_range[peak_indices]
    print("Max E:",max_E)

    plt.semilogy(E_range,PES,color = "k",label = "PES")

    for i in peak_indices:
        plt.scatter(E_range[i],PES[i],color = "b",label =  f"E = {E_range[i]}")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.175), ncol=3)
    plt.savefig("images/peak.png")
    plt.clf()

PAD = np.load("PES_files/PAD.npy")

k_vals = np.real(PAD[0,:])
theta_vals = np.real(PAD[1,:])
phi_vals = np.real(PAD[2,:])
pad_vals = np.real(PAD[3,:])


kx_vals = k_vals* np.sin(theta_vals) * np.cos(phi_vals)
ky_vals = k_vals * np.sin(theta_vals) * np.sin(phi_vals)
kz_vals = k_vals * np.cos(theta_vals)

max = np.max(np.real(pad_vals))
min = np.max(np.real(pad_vals))*10**-6

if SLICE == "XY":
    fig, ax = plt.subplots()  # Create a figure and axes
    viridis = plt.get_cmap('viridis')  # Get the viridis colormap
    smallest_color = viridis(0)  # Get the smallest color from the colormap

    print("TEST")

    ax.set_facecolor(smallest_color)
    if LOG:
        sc = ax.scatter(kx_vals, ky_vals, c=pad_vals, cmap="viridis",norm=mcolors.LogNorm(vmin=min,vmax=max))
    else:
        sc = ax.scatter(kx_vals, ky_vals, c=pad_vals, cmap="viridis")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    fig.colorbar(sc, ax=ax)
    plt.savefig("images/PAD.png")
elif SLICE == "XZ":
    fig, ax = plt.subplots()  # Create a figure and axes
    viridis = plt.get_cmap('viridis')  # Get the viridis colormap
    smallest_color = viridis(0)  # Get the smallest color from the colormap


    ax.set_facecolor(smallest_color)
    if LOG:
        sc = ax.scatter(kz_vals, kx_vals, c=pad_vals, cmap="viridis",norm=mcolors.LogNorm(vmin=min,vmax=max))
    else:
        sc = ax.scatter(kz_vals, kx_vals, c=pad_vals, cmap="viridis")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("kz")
    ax.set_ylabel("kx")
    fig.colorbar(sc, ax=ax)
    print("TEST")
    plt.savefig("images/PAD.png")






import numpy as np
import matplotlib.pyplot as plt

# Laser/Species Conditions

Ip = 0.5
N = 10
wavelength = 400
w = 0.057 * 800/wavelength
I = 2.0e14 / 3.51E16

tau = 2*np.pi/w
t = np.linspace(0,tau,1000)
envelope = np.sin(np.pi*t/tau)**2
I_profile = I * envelope
Up = I_profile/(4*w**2)

E = 0.240

def print_direct_cutoff():
    energy = 2*np.max(Up)
    print(f"Direct cutoff energy is {energy}")
def print_rescattering_cutoff():
    energy = 10*np.max(Up)
    print(f"Rescattering cutoff energy is {energy}")
def print_keldysh_parameter():
    gamma = np.sqrt(Ip/(2*np.max(Up)))
    print(f"Keldysh parameter is {gamma}")
    return gamma
def print_intensity_parameter():
    z = np.max(Up)/w
    print(f"Intensity parameter is {z}")
    


def find_photons(E):
    n = (E+Ip+Up)/w
    return n
def plot_channels(n,gamma):
    channel_indices = np.where(np.diff(np.floor(n)))[0]

    for idx in channel_indices:
        x_int = t[idx]
        y_int = n[idx]
        plt.plot([x_int, x_int], [np.min(n), y_int], color='k')
        plt.axhline(y_int,linestyle="dashed")

    plt.plot(t,n)
    print("Largest Channel:",np.max(n))
    plt.ylabel("Number of Photons")
    plt.xlabel("Time (a.u.)")
    plt.title(rf"Channel Closings: $\gamma$ = {round(gamma,3)}")
    plt.savefig("images/channels.png")
    plt.clf()

     
if __name__ == "__main__":
    print_direct_cutoff()
    print("\n")
    print_rescattering_cutoff()
    print("\n")
    gamma = print_keldysh_parameter()
    print("\n")
    print_intensity_parameter()
    print("\n")
    n = find_photons(E)
    plot_channels(n,gamma)
    print("Done")

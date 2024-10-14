import numpy as np
import matplotlib.pyplot as plt

# Laser/Species Conditions

Ip = 0.5
N = 10
wavelength = 800
w = 0.057 * 800/wavelength
I = 2e14 / 3.51E16

tau = 2*np.pi/w
t = np.linspace(0,N*tau,1000)
envelope = np.sin(np.pi*t/tau)**2
I_profile = I * envelope
Up = I_profile/(4*w**2)

def print_direct_cutoff():
    energy = Ip + np.max(Up)
    print(f"Direct cutoff energy is {energy}")
def print_rescattering_cutoff():
    energy = Ip + 10 * np.max(Up)
    print(f"Rescattering cutoff energy is {energy}")
def print_keldysh_parameter():
    gamma = np.sqrt(Ip/(2*np.max(Up)))
    print(f"Keldysh parameter is {gamma}")
def print_intensity_parameter():
    z = np.max(Up)/w
    print(f"Intensity parameter is {z}")


def find_photons(E):
    n = (E+Ip+Up)/w
    return n
def plot_channels(n):
    channel_indices = np.where(np.diff(np.floor(n)))[0]

    for idx in channel_indices:
        x_int = t[idx]
        y_int = n[idx]
        plt.plot([x_int, x_int], [np.min(n), y_int], color='k')
        plt.axhline(y_int,linestyle="dashed")

    plt.plot(t,n)
    plt.savefig("images/channels.png")
    plt.clf()
            
if __name__ == "__main__":
    print_direct_cutoff()
    print_rescattering_cutoff()
    print_keldysh_parameter()
    print_intensity_parameter()
    n = find_photons(1.4)
    plot_channels(n)
    print("Done")
import numpy as np 
import matplotlib.pyplot as plt

##########################################################################################################################

# Ip = -0.5791546178 # Ionization potential of species in atomic units
Ip = -0.5
N = 10 # Numbe08r of cycles of laser pulse 


lam = 1200
w = 0.057 * 800/lam
#w = 0.057# Central Frequency of laser pulse in atomic units
I = 2e14 / 3.51E16 # Intensity of laser pulse in atomic units
E = 0.48 # Energy in atomic units (usually at ATI peak)
#E = 0.608

tau = 2*np.pi/w # Period of laser pulse in atomic units
t = np.linspace(0,tau,1000) # Time array
envelope = np.sin(np.pi*t/tau)**2 # Envelope function, usually Sin2
I_profile = I * envelope # Intensity profile of laser pulse
Up = I_profile/(4*w**2) # Pondermotive energy in atomic units

############################################################################################################################

def findPhotons(E,Ip,Up,w):
    n = (E-Ip+Up)/w 
    return n

n = findPhotons(E,Ip,Up,w)
channel_indices = np.where(np.diff(np.floor(n)))[0]

for idx in channel_indices:
    x_int = t[idx]
    y_int = n[idx]
    plt.plot([x_int, x_int], [np.min(n), y_int], color='k')
    plt.axhline(y_int,linestyle="dashed")

plt.plot(t,n)
plt.savefig("images/channels.png")
plt.clf()
#############################################################################################################################

def findIntensityParameter(Up,w):
    z = Up/w
    return z

z = findIntensityParameter(Up,w)
plt.plot(t,z)
print(f"Max value of Up/w is {np.max(z)}")
plt.savefig("images/z.png")
plt.clf()

def findKeldyshParameter(Ip,Up):
    gamma = np.sqrt(-Ip/(2*np.max(Up)))
    return gamma

gamma = findKeldyshParameter(Ip,Up)
print(f"Keldysh parameter is {gamma}")

##########################################################################################################################################


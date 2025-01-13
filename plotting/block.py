import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize
import json as json
import sys as sys

file_path = "input.json"
with open(file_path, 'r') as file:
    input_data = json.load(file)

pyramid_array = np.load("TDSE_files/pyramid_array.npy")

LOG = "LOG" in sys.argv


fig, ax = plt.subplots(figsize=(10, 8))

norm = LogNorm() if LOG else Normalize()

cax = ax.imshow(pyramid_array[::-1], cmap='inferno', interpolation='nearest', norm=norm)  
ax.set_xlabel('m')
ax.set_ylabel('l')
ax.set_xticks([0, input_data["lm"]["lmax"], 2 * input_data["lm"]["lmax"]])  
ax.set_xticklabels([-input_data["lm"]["lmax"], 0, input_data["lm"]["lmax"]])
ax.set_yticks([0, input_data["lm"]["lmax"]])  
ax.set_yticklabels([input_data["lm"]["lmax"], 0])  
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False) 

fig.colorbar(cax, ax=ax, shrink=0.5)
ax.set_title('Heatmap of Probabilities for l and m Values')
fig.savefig("images/pyramid.png")
fig.clf()



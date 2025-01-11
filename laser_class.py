import matplotlib.pyplot as plt
import numpy as np

from petsc4py import PETSc

rank = PETSc.COMM_WORLD.rank
size = PETSc.COMM_WORLD.size

class laser:
    def __init__(self,sim):
        pass

    def envFunc(self,t,sim):
        env = sim.input_params["laser"]["A_0"] * np.sin(sim.input_params["laser"]["w"] * t/(2*sim.input_params["box"]["N"]))**2
        return env
      
    def Ax(self,t,sim):
        return np.where(t<=sim.input_params["box"]["time_size"],self.envFunc(t,sim) * 1 / np.sqrt(1 + sim.input_params["laser"]["ell"]**2) * (sim.input_params["laser"]["polarization"][0] * np.sin(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"]) + sim.input_params["laser"]["ell"] * sim.input_params["laser"]["ellipticity"][0] * np.cos(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"])),0.0)
        
    def Ay(self,t,sim):
        return np.where(t<=sim.input_params["box"]["time_size"],self.envFunc(t,sim) * 1 / np.sqrt(1 + sim.input_params["laser"]["ell"]**2) * (sim.input_params["laser"]["polarization"][1] * np.sin(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"]) + sim.input_params["laser"]["ell"] * sim.input_params["laser"]["ellipticity"][1] * np.cos(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"])),0.0)
    
    def Az(self,t,sim):
        return np.where(t<=sim.input_params["box"]["time_size"],self.envFunc(t,sim) * 1 / np.sqrt(1 + sim.input_params["laser"]["ell"]**2) * (sim.input_params["laser"]["polarization"][2] * np.sin(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"]) + sim.input_params["laser"]["ell"] * sim.input_params["laser"]["ellipticity"][2] * np.cos(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"])),0.0)


    def plotPulse(self,sim):
        if rank == 0:
            t_start,t_end,Nt = sim.input_params["box"]["t_range"]
            t_post_start,t_post_end,Nt_post = sim.input_params["box"]["t_post_range"]

            

            t = np.linspace(t_start,t_end,int(Nt))
            t_post = np.linspace(t_post_start,t_post_end,int(Nt_post))
            t_total = np.concatenate((t,t_post))
            plt.figure()  # Explicitly create a new figure
            plt.plot(t_total, self.Ax(t_total, sim), label="X")
            plt.plot(t_total, self.Ay(t_total, sim), label="Y")
            plt.plot(t_total, self.Az(t_total, sim), label="Z")
            plt.legend()
            plt.savefig("images/laser.png")
            plt.close()  # Close the figure to ensure it doesn't interfere with subsequent plots
        return True

            
    
        


import matplotlib.pyplot as plt
import numpy as np

from petsc4py import PETSc

rank = PETSc.COMM_WORLD.rank
size = PETSc.COMM_WORLD.size

class laser:
    def __init__(self,sim):

        def envFunc(t):
            env = sim.input_params["laser"]["A_0"] * np.sin(sim.input_params["laser"]["w"] * t/(2*sim.input_params["box"]["N"]))**2
            return env
        self.envFunc = envFunc

        
        
       
         
        def Ax(t,sim):
            return np.where(t<=sim.input_params["box"]["time_size"],self.envFunc(t) * 1 / np.sqrt(1 + sim.input_params["laser"]["ell"]**2) * (sim.input_params["laser"]["polarization"][0] * np.sin(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"]) + sim.input_params["laser"]["ell"] *  sim.input_params["laser"]["ellipticity"][0] * np.cos(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"])),0.0)
            
        def Ay(t,sim):
            return np.where(t<=sim.input_params["box"]["time_size"],self.envFunc(t) * 1 / np.sqrt(1 + sim.input_params["laser"]["ell"]**2) * (sim.input_params["laser"]["polarization"][1] * np.sin(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"]) + sim.input_params["laser"]["ell"] *  sim.input_params["laser"]["ellipticity"][1] * np.cos(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"])),0.0)
        
        def Az(t,sim):
            return np.where(t<=sim.input_params["box"]["time_size"],self.envFunc(t) * 1 / np.sqrt(1 + sim.input_params["laser"]["ell"]**2) * (sim.input_params["laser"]["polarization"][2] * np.sin(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"]) + sim.input_params["laser"]["ell"] *  sim.input_params["laser"]["ellipticity"][2] * np.cos(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"])),0.0)
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az

    def plotPulse(self,sim):
        t_start,t_end,Nt = sim.input_params["box"]["t_range"]
        t_post_start,t_post_end,Nt_post = sim.input_params["box"]["t_post_range"]

        t = np.linspace(t_start,t_end,int(Nt))
        t_post = np.linspace(t_post_start,t_post_end,int(Nt_post))
        total_time = np.concatenate((t,t_post))



        if rank == 0:
            plt.plot(total_time,self.Ax(total_time,sim),label = "X")
            plt.plot(total_time,self.Ay(total_time,sim),label = "Y")
            plt.plot(total_time,self.Az(total_time,sim),label = "Z")
            plt.legend()
            plt.savefig("images/laser.png")
            plt.clf()


        
    
        
        




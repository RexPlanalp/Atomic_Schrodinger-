import matplotlib.pyplot as plt
import numpy as np

from petsc4py import PETSc

rank = PETSc.COMM_WORLD.rank
size = PETSc.COMM_WORLD.size

class laser:
    # def __init__(self,sim):
    #     pass

    # def envFunc(self,t,sim):
    #     env = sim.input_params["laser"]["A_0"] * np.sin(sim.input_params["laser"]["w"] * t/(2*sim.input_params["box"]["N"]))**2
    #     return env
      
    # def Ax(self,t,sim):
    #     return np.where(t<=sim.input_params["box"]["time_size"],self.envFunc(t,sim) * 1 / np.sqrt(1 + sim.input_params["laser"]["ell"]**2) * (sim.input_params["laser"]["polarization"][0] * np.sin(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"]) + sim.input_params["laser"]["ell"] * sim.input_params["laser"]["ellipticity"][0] * np.cos(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"])),0.0)
        
    # def Ay(self,t,sim):
    #     return np.where(t<=sim.input_params["box"]["time_size"],self.envFunc(t,sim) * 1 / np.sqrt(1 + sim.input_params["laser"]["ell"]**2) * (sim.input_params["laser"]["polarization"][1] * np.sin(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"]) + sim.input_params["laser"]["ell"] * sim.input_params["laser"]["ellipticity"][1] * np.cos(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"])),0.0)
    
    # def Az(self,t,sim):
    #     return np.where(t<=sim.input_params["box"]["time_size"],self.envFunc(t,sim) * 1 / np.sqrt(1 + sim.input_params["laser"]["ell"]**2) * (sim.input_params["laser"]["polarization"][2] * np.sin(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"]) + sim.input_params["laser"]["ell"] * sim.input_params["laser"]["ellipticity"][2] * np.cos(sim.input_params["laser"]["w"] * t+np.pi*sim.input_params["laser"]["CEP"])),0.0)


    # def plotPulse(self,sim):
    #     if rank == 0:
    #         t_start,t_end,Nt = sim.input_params["box"]["t_range"]
    #         t_post_start,t_post_end,Nt_post = sim.input_params["box"]["t_post_range"]

            

    #         t = np.linspace(t_start,t_end,int(Nt))
    #         t_post = np.linspace(t_post_start,t_post_end,int(Nt_post))
    #         t_total = np.concatenate((t,t_post))
    #         plt.figure()  # Explicitly create a new figure
    #         plt.plot(t_total, self.Ax(t_total, sim), label="X")
    #         plt.plot(t_total, self.Ay(t_total, sim), label="Y")
    #         plt.plot(t_total, self.Az(t_total, sim), label="Z")
    #         plt.legend()
    #         plt.savefig("images/laser.png")
    #         plt.close()  # Close the figure to ensure it doesn't interfere with subsequent plots

    #         if sim.input_params["laser"]["components"][0] and sim.input_params["laser"]["components"][1]:
    #             plt.plot(self.Ax(t_total,sim),self.Ay(t_total,sim))
    #             plt.savefig("images/laser_xy.png")
    #             plt.clf()
    #     return True
    def __init__(self,sim):
        

        import json
        with open("input.json", "r") as f:
            input_par = json.load(f)

        w = input_par["laser"]["w"]
        I = input_par["laser"]["I"]/3.51E16
        self.CEP = input_par["laser"]["CEP"]*np.pi
        E_0 = np.sqrt(I)
        self.A_0 = E_0/w

        
        N = input_par["box"]["N"]
        N_post = input_par["box"]["N_post"]

        ell = -input_par["laser"]["ell"]

       
        
        def envFunc(t):
            env = self.A_0 * np.sin(w * t/(2*N))**2
            return env
            
        polarization = input_par["laser"]["polarization"]
        polarization /= np.linalg.norm(polarization)

        poynting = input_par["laser"]["poynting"]
        poynting /= np.linalg.norm(poynting)

        ellipticity_Vector = np.cross(polarization, poynting) 
        ellipticity_Vector /= np.linalg.norm(ellipticity_Vector)

        self.components  = [(1 if polarization[i] != 0 or ell * ellipticity_Vector[i] != 0 else 0) for i in range(3)]



        time_size = N * 2*np.pi / w
        post_time_size = N_post* 2*np.pi / w

        dt = input_par["box"]["time_spacing"]

        Nt = int(np.rint(time_size/ dt)) 
        Nt_post = int(np.rint(post_time_size/ dt)) 

        t = np.linspace(0,time_size,Nt)
        t_post = np.linspace(time_size,time_size+post_time_size,Nt_post)

        self.total_time = np.concatenate((t,t_post))
        self.t = t
        self.t_post = t_post

        def Ax(t,sim):
            return np.where(t<=time_size,envFunc(t) * 1 / np.sqrt(1 + ell**2) * (polarization[0] * np.sin(w * t+self.CEP) - ell * ellipticity_Vector[0] * np.cos(w * t+self.CEP)),0.0)
            
        def Ay(t,sim):
            return np.where(t<=time_size,envFunc(t) * 1 / np.sqrt(1 + ell**2) * (polarization[1] * np.sin(w * t+self.CEP) - ell * ellipticity_Vector[1] * np.cos(w * t+self.CEP)),0.0)
        
        def Az(t,sim):
            return np.where(t<=time_size,envFunc(t) * 1 / np.sqrt(1 + ell**2) * (polarization[2] * np.sin(w * t+self.CEP) - ell * ellipticity_Vector[2] * np.cos(w * t+self.CEP)),0.0)
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az
                

    def plotPulse(self,sim):
        if rank == 0:
            plt.plot(self.total_time,self.Ax(self.total_time,sim),label = "X")
            plt.plot(self.total_time,self.Ay(self.total_time,sim),label = "Y")
            plt.plot(self.total_time,self.Az(self.total_time,sim),label = "Z")
            plt.legend()
            plt.savefig("images/laser.png")
            plt.clf()


        if rank == 0 and self.components[0] and self.components[1]:
            plt.plot(self.Ax(self.total_time,sim),self.Ay(self.total_time,sim))
            plt.savefig("images/laser_xy.png")
            plt.clf()
    
        
        




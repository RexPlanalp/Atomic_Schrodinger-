import matplotlib.pyplot as plt
import numpy as np
import sys

from petsc4py import PETSc
rank = PETSc.COMM_WORLD.rank

sys.path.append('/users/becker/dopl4670/Research/Atomic_Schrodinger/Common')
import utility

class laser:
    def __init__(self,TDSEInstance):


        w = TDSEInstance.parameters["lasers"]["w"]
        I = TDSEInstance.parameters["lasers"]["I"]/3.51E16
        E_0 = np.sqrt(I)
        self.A_0 = E_0/w

        
        N = TDSEInstance.parameters["box"]["N"]
        N_post = TDSEInstance.parameters["box"]["N_post"]

        ell = -TDSEInstance.parameters["lasers"]["ell"]

       
        
        def envFunc(t):
            env = self.A_0 * np.sin(w * t/(2*N))**2
            return env
            
        polarization = TDSEInstance.parameters["lasers"]["polarization"]
        polarization /= np.linalg.norm(polarization)

        poynting = TDSEInstance.parameters["lasers"]["poynting"]
        poynting /= np.linalg.norm(poynting)

        ellipticity_Vector = np.cross(polarization, poynting) 
        ellipticity_Vector /= np.linalg.norm(ellipticity_Vector)

        self.components  = [(1 if polarization[i] != 0 or ell * ellipticity_Vector[i] != 0 else 0) for i in range(3)]



        time_size = N * 2*np.pi / w
        post_time_size = N_post* 2*np.pi / w

        dt = TDSEInstance.parameters["box"]["time_spacing"]

        Nt = int(np.rint(time_size/ dt)) 
        Nt_post = int(np.rint(post_time_size/ dt)) 

        t = np.linspace(0,time_size,Nt)
        t_post = np.linspace(time_size,time_size+post_time_size,Nt_post)

        self.total_time = np.concatenate((t,t_post))
        self.t = t
        self.t_post = t_post

        def Ax(t):
            return np.where(t<=time_size,envFunc(t) * 1 / np.sqrt(1 + ell**2) * (polarization[0] * np.sin(w * t) - ell * ellipticity_Vector[0] * np.cos(w * t)),0.0)
            
        def Ay(t):
            return np.where(t<=time_size,envFunc(t) * 1 / np.sqrt(1 + ell**2) * (polarization[1] * np.sin(w * t) - ell * ellipticity_Vector[1] * np.cos(w * t)),0.0)
        
        def Az(t):
            return np.where(t<=time_size,envFunc(t) * 1 / np.sqrt(1 + ell**2) * (polarization[2] * np.sin(w * t) - ell * ellipticity_Vector[2] * np.cos(w * t)),0.0)
        self.Ax_func = Ax
        self.Ay_func = Ay
        self.Az_func = Az
                

    def plotPulse(self):
        if rank == 0:
            plt.plot(self.total_time,self.Ax_func(self.total_time),label = "X")
            plt.plot(self.total_time,self.Ay_func(self.total_time),label = "Y")
            plt.plot(self.total_time,self.Az_func(self.total_time),label = "Z")
            plt.legend()
            plt.savefig("images/laser.png")
            plt.clf()

            np.save("TDSE_files/t_total.npy",self.total_time)
    
        
        


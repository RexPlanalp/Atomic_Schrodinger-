import numpy as np
import json as json
import pprint as pprint
import h5py as h5py
import os as os

from collections.abc import Mapping
from collections import deque
from petsc4py import PETSc
import petsc4py.PETSc
from slepc4py import SLEPc
import petsc4py

import simulation_class as simulation

rank = PETSc.COMM_WORLD.rank
size = PETSc.COMM_WORLD.size

class tdse(simulation.simulation):
    def __init__(self,input_file):
        super().__init__(input_file)
    
    def readInState(self):
        block_index = self.input_params["lm"]["lm_to_block"][(self.input_params["state"][1], self.input_params["state"][2])]
        psi_initial = PETSc.Vec().createMPI(self.input_params["splines"]["n_basis"]*self.input_params["lm"]["n_blocks"],comm = PETSc.COMM_WORLD)
        
        with h5py.File(f'TISE_files/{self.input_params["box"]["potential_func"].__name__}.h5', 'r') as f:
            data = f[f'/Psi_{self.input_params["state"][0]}_{self.input_params["state"][1]}'][:]
            real_part = data[:,0]
            imaginary_part = data[:,1]
            total = real_part + 1j*imaginary_part

        global_indices = np.array(range(self.input_params["splines"]["n_basis"]))+block_index*self.input_params["splines"]["n_basis"]
        global_indices = global_indices.astype("int32")
        psi_initial.setValues(global_indices,total)
        psi_initial.assemble()
        self.state = psi_initial

        #####################################
        # TESTING
        # psi_test = PETSc.Vec().createMPI(self.input_params["splines"]["n_basis"]*self.input_params["lm"]["n_blocks"],comm = PETSc.COMM_WORLD)
        
        # with h5py.File(f'../../qprop_circ_sample/TISE_files/{self.input_params["box"]["potential_func"].__name__}.h5', 'r') as f:
        #     data = f[f'/Psi_{self.input_params["state"][0]}_{self.input_params["state"][1]}'][:]
        #     real_part = data[:,0]
        #     imaginary_part = data[:,1]
        #     total = real_part + 1j*imaginary_part

        # psi_test.setValues(global_indices,total)
        # psi_test.assemble()
        
        # diff = psi_initial.copy()
        # diff.axpy(-1,psi_test)
        # diff_val = diff.norm(PETSc.NormType.FROBENIUS)
        # PETSc.Sys.Print(f"Norm of difference: {diff_val}",comm=PETSc.COMM_WORLD)
        ###############################################################


        return True

    def constructInteraction(self,basis):
        self.hamiltonians = [0]*3

        def _H_der_R_element(x,i,j,knots,order):
            return (basis.B(i, self.input_params["splines"]["order"], x, knots)*basis.dB(j, self.input_params["splines"]["order"], x, knots))
        H_der_R = PETSc.Mat().createAIJ([self.input_params["splines"]["n_basis"],self.input_params["splines"]["n_basis"]],comm = PETSc.COMM_WORLD,nnz = 2*(self.input_params["splines"]["order"]-1)+1)
        istart,iend = H_der_R.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(self.input_params["splines"]["n_basis"]):
                if np.abs(i-j)>=self.input_params["splines"]["order"]:
                    continue
                H_element = basis.integrate(_H_der_R_element,i,j,self.input_params["splines"]["order"],basis.knots)
                H_der_R.setValue(i,j,H_element)
        PETSc.COMM_WORLD.barrier()
        H_der_R.assemble()

        def _H_inv_R_element(x,i,j,knots,order):
            return (basis.B(i, self.input_params["splines"]["order"], x, knots)*basis.B(j, self.input_params["splines"]["order"], x, knots)/ (x+1E-25))
        H_inv_R =  PETSc.Mat().createAIJ([self.input_params["splines"]["n_basis"],self.input_params["splines"]["n_basis"]],comm = PETSc.COMM_WORLD,nnz = 2*(self.input_params["splines"]["order"]-1)+1)
        istart,iend = H_inv_R.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(self.input_params["splines"]["n_basis"]):
                if np.abs(i-j)>=self.input_params["splines"]["order"]:
                    continue
                H_element = basis.integrate(_H_inv_R_element,i,j,self.input_params["splines"]["order"],basis.knots)
                H_inv_R.setValue(i,j,H_element)
        PETSc.COMM_WORLD.barrier()
        H_inv_R.assemble()


        if self.input_params["laser"]["components"][0] or self.input_params["laser"]["components"][1]:
            def a(l,m):
                f1 = np.sqrt((l+m)/((2*l+1)*(2*l-1)))
                f2 = -m * np.sqrt(l+m-1) - np.sqrt((l-m)*(l*(l-1)- m*(m-1)))
                return f1*f2
            def atilde(l,m):
                f1 = np.sqrt((l-m)/((2*l+1)*(2*l-1)))
                f2 = -m * np.sqrt(l-m-1) + np.sqrt((l+m)*(l*(l-1)- m*(m+1)))
                return f1*f2

            def b(l,m):
                return -atilde(l+1,m-1)
            def btilde(l,m):
                return -a(l+1,m+1)

            def c(l,m):
                return dtilde(l-1,m-1)
            def ctilde(l,m):
                return d(l-1,m+1)

            def d(l,m):
                f1 = np.sqrt((l-m+1)*(l-m+2))
                f2 = np.sqrt((2*l+1)*(2*l+3))
                return f1/f2
            def dtilde(l,m):
                return d(l,-m)

            H_XY_lm_1 = PETSc.Mat().createAIJ([self.input_params["lm"]["n_blocks"],self.input_params["lm"]["n_blocks"]],comm = PETSc.COMM_WORLD,nnz = 2)
            istart,iend = H_XY_lm_1.getOwnershipRange()
            for i in range(istart,iend):
                l,m = self.input_params["lm"]["block_to_lm"][i]
                for j in range(self.input_params["lm"]["n_blocks"]):
                    lprime,mprime = self.input_params["lm"]["block_to_lm"][j]
                    if (l == lprime+1) and (m == mprime+1):
                        H_XY_lm_1.setValue(i,j,a(l,m))
                    elif (l == lprime-1) and (m == mprime+1):
                        H_XY_lm_1.setValue(i,j,b(l,m))
            PETSc.COMM_WORLD.barrier()
            H_XY_lm_1.assemble()

            H_XY_lm_2 = PETSc.Mat().createAIJ([self.input_params["lm"]["n_blocks"],self.input_params["lm"]["n_blocks"]],comm = PETSc.COMM_WORLD,nnz = 2)
            istart,iend = H_XY_lm_2.getOwnershipRange()
            for i in range(istart,iend):
                l,m = self.input_params["lm"]["block_to_lm"][i]
                for j in range(self.input_params["lm"]["n_blocks"]):
                    lprime,mprime = self.input_params["lm"]["block_to_lm"][j]
                    if (l == lprime+1) and (m == mprime+1):
                        H_XY_lm_2.setValue(i,j,c(l,m))
                    elif (l == lprime-1) and (m == mprime+1):
                        H_XY_lm_2.setValue(i,j,-d(l,m))
            PETSc.COMM_WORLD.barrier()
            H_XY_lm_2.assemble()


            term1 = self.kron(H_XY_lm_1,H_inv_R,PETSc.COMM_WORLD,2*(2*(self.input_params["splines"]["order"]-1) + 1))
            term2 = self.kron(H_XY_lm_2,H_der_R,PETSc.COMM_WORLD,2*(2*(self.input_params["splines"]["order"]-1) + 1))

            term1.axpy(1,term2)
            term1.scale(1j/2)
            term2.destroy()
            H_XY_lm_1.destroy()
            H_XY_lm_2.destroy()

            self.hamiltonians[0] = term1


            H_XY_lm_3 = PETSc.Mat().createAIJ([self.input_params["lm"]["n_blocks"],self.input_params["lm"]["n_blocks"]],comm = PETSc.COMM_WORLD,nnz = 2)
            istart,iend = H_XY_lm_3.getOwnershipRange()
            for i in range(istart,iend):
                l,m = self.input_params["lm"]["block_to_lm"][i]
                for j in range(self.input_params["lm"]["n_blocks"]):
                    lprime,mprime = self.input_params["lm"]["block_to_lm"][j]
                    if (l == lprime+1) and (m == mprime-1):
                        H_XY_lm_3.setValue(i,j,atilde(l,m))
                    elif (l == lprime-1) and (m == mprime-1):
                        H_XY_lm_3.setValue(i,j,btilde(l,m))
            PETSc.COMM_WORLD.barrier()
            H_XY_lm_3.assemble()

            H_XY_lm_4 = PETSc.Mat().createAIJ([self.input_params["lm"]["n_blocks"],self.input_params["lm"]["n_blocks"]],comm = PETSc.COMM_WORLD,nnz = 2)
            istart,iend = H_XY_lm_4.getOwnershipRange()
            for i in range(istart,iend):
                l,m = self.input_params["lm"]["block_to_lm"][i]
                for j in range(self.input_params["lm"]["n_blocks"]):
                    lprime,mprime = self.input_params["lm"]["block_to_lm"][j]
                    if (l == lprime+1) and (m == mprime-1):
                        H_XY_lm_4.setValue(i,j,-ctilde(l,m))
                    elif (l == lprime-1) and (m == mprime-1):
                        H_XY_lm_4.setValue(i,j,dtilde(l,m))
            H_XY_lm_4.assemble()

            term3 = self.kron(H_XY_lm_3,H_inv_R,PETSc.COMM_WORLD,2*(2*(self.input_params["splines"]["order"]-1) + 1))
            term4 = self.kron(H_XY_lm_4,H_der_R,PETSc.COMM_WORLD,2*(2*(self.input_params["splines"]["order"]-1) + 1))

            term3.axpy(1,term4)
            term3.scale(1j/2)
            term4.destroy()
            H_XY_lm_3.destroy()
            H_XY_lm_4.destroy()
            
            self.hamiltonians[1] = term3
            
        if self.input_params["laser"]["components"][2]:
            def clm(l,m):
                return -1j*np.sqrt(((l+1)**2 - m**2)/((2*l+1)*(2*l+3)))
            def dlm(l,m):
                return -1j*np.sqrt(((l)**2 - m**2)/((2*l-1)*(2*l+1)))
            
            H_Z_lm_1 = PETSc.Mat().createAIJ([self.input_params["lm"]["n_blocks"],self.input_params["lm"]["n_blocks"]],comm = PETSc.COMM_WORLD,nnz = 2)
            istart,iend = H_Z_lm_1.getOwnershipRange()
            for i in range(istart,iend):
                l,m = self.input_params["lm"]["block_to_lm"][i]
                for j in range(self.input_params["lm"]["n_blocks"]):
                    lprime,mprime = self.input_params["lm"]["block_to_lm"][j]
                    if (l == lprime+1) and (m == mprime):
                        H_Z_lm_1.setValue(i,j,dlm(l,m))
                    elif (l == lprime-1) and (m == mprime):
                        H_Z_lm_1.setValue(i,j,clm(l,m))
            PETSc.COMM_WORLD.barrier()
            H_Z_lm_1.assemble()

            H_Z_lm_2 = PETSc.Mat().createAIJ([self.input_params["lm"]["n_blocks"],self.input_params["lm"]["n_blocks"]],comm = PETSc.COMM_WORLD,nnz = 2)
            istart,iend = H_Z_lm_2.getOwnershipRange()
            for i in range(istart,iend):
                l,m = self.input_params["lm"]["block_to_lm"][i]
                for j in range(self.input_params["lm"]["n_blocks"]):
                    lprime,mprime = self.input_params["lm"]["block_to_lm"][j]
                    if (l == lprime+1) and (m == mprime):
                        H_Z_lm_2.setValue(i,j,-(l)*dlm(l,m))
                    elif (l == lprime-1) and (m == mprime):
                        H_Z_lm_2.setValue(i,j,(l+1)*clm(l,m))
            PETSc.COMM_WORLD.barrier()
            H_Z_lm_2.assemble()

            H_Z_1 = self.kron(H_Z_lm_1,H_der_R,PETSc.COMM_WORLD,2*(2*(self.input_params["splines"]["order"]-1)+1))
            H_Z_2 = self.kron(H_Z_lm_2,H_inv_R,PETSc.COMM_WORLD,2*(2*(self.input_params["splines"]["order"]-1)+1))
            H_Z_lm_1.destroy()
            H_Z_lm_2.destroy()

            H_Z_1.axpy(1.0,H_Z_2)

            self.hamiltonians[2] = H_Z_1

    def constructAtomic(self):
        

        if os.path.exists('TISE_files/K.bin'):
            K = PETSc.Mat().createAIJ([self.input_params["splines"]["n_basis"],self.input_params["splines"]["n_basis"]],comm = PETSc.COMM_WORLD,nnz = 2*(self.input_params["splines"]["order"]-1)+1)
            K_viewer = PETSc.Viewer().createBinary('TISE_files/K.bin', 'r')
            K.load(K_viewer)
            K_viewer.destroy()
            K.assemble()
        
        if os.path.exists('TISE_files/S.bin'):
            S = PETSc.Mat().createAIJ([self.input_params["splines"]["n_basis"],self.input_params["splines"]["n_basis"]],comm = PETSc.COMM_WORLD,nnz = 2*(self.input_params["splines"]["order"]-1)+1)
            S_viewer = PETSc.Viewer().createBinary('TISE_files/S.bin', 'r')
            S.load(S_viewer)
            S_viewer.destroy()
            S.assemble()

        if os.path.exists('TISE_files/Invr2.bin'):
            Invr2 = PETSc.Mat().createAIJ([self.input_params["splines"]["n_basis"],self.input_params["splines"]["n_basis"]],comm = PETSc.COMM_WORLD,nnz = 2*(self.input_params["splines"]["order"]-1)+1)
            Invr2_viewer = PETSc.Viewer().createBinary('TISE_files/Invr2.bin', 'r')
            Invr2.load(Invr2_viewer)
            Invr2_viewer.destroy()
            Invr2.assemble()
        
        if os.path.exists('TISE_files/V.bin'):
            V = PETSc.Mat().createAIJ([self.input_params["splines"]["n_basis"],self.input_params["splines"]["n_basis"]],comm = PETSc.COMM_WORLD,nnz = 2*(self.input_params["splines"]["order"]-1)+1)
            V_viewer = PETSc.Viewer().createBinary('TISE_files/V.bin', 'r')
            V.load(V_viewer)
            V_viewer.destroy()
            V.assemble()

        I = PETSc.Mat().createAIJ([self.input_params["lm"]["n_blocks"],self.input_params["lm"]["n_blocks"]],comm = PETSc.COMM_WORLD)
        istart,iend = I.getOwnershipRange()
        for i in range(istart,iend):
            I.setValue(i,i,1)
        PETSc.COMM_WORLD.barrier()
        I.assemble()

        S_total = self.kron(I,S,PETSc.COMM_WORLD,2*(self.input_params["splines"]["order"]-1)+1)
        
        I.destroy()
        S.destroy()

        H_total= PETSc.Mat().createAIJ([self.input_params["lm"]["n_blocks"]*self.input_params["splines"]["n_basis"],self.input_params["lm"]["n_blocks"]*self.input_params["splines"]["n_basis"]],comm = PETSc.COMM_WORLD,nnz = 2*(self.input_params["splines"]["order"]-1)+1)
        H_total.assemble()

        for l in range(self.input_params["lm"]["lmax"]+1):
            V_l = Invr2.copy()
            V_l.scale(l * (l + 1) / 2)
            V_l.axpy(1.0, V)

            H_l = K.copy()
            H_l.axpy(1.0, V_l)

            


            block_indices = []
            for (lprime,mprime),block_idx in self.input_params["lm"]["lm_to_block"].items():
                if lprime == l:
                    block_indices.append(block_idx)
                
            
            partial_I = PETSc.Mat().createAIJ([self.input_params["lm"]["n_blocks"],self.input_params["lm"]["n_blocks"]],comm = PETSc.COMM_WORLD,nnz = 1)
            istart,iend = partial_I.getOwnershipRange()
            for i in range(istart,iend):
                if i in block_indices:
                    partial_I.setValue(i,i,1)
            PETSc.COMM_WORLD.barrier()
            partial_I.assemble()

            partial_H = self.kron(partial_I,H_l,PETSc.COMM_WORLD,2*(self.input_params["splines"]["order"]-1) +1)
            H_total.axpy(1,partial_H)

            partial_I.destroy()
            partial_H.destroy()
            V_l.destroy()
            H_l.destroy()
        K.destroy()
        Invr2.destroy()
        V.destroy()

        #####################################
        # TESTING

        # if os.path.exists('../../qprop_circ_sample/TISE_files/S.bin'):
        #     S_test = PETSc.Mat().createAIJ([self.input_params["splines"]["n_basis"]*self.input_params["lm"]["n_blocks"],self.input_params["splines"]["n_basis"]*self.input_params["lm"]["n_blocks"]],comm = PETSc.COMM_WORLD,nnz = 2*(self.input_params["splines"]["order"]-1)+1)
        #     S_viewer = PETSc.Viewer().createBinary('../../qprop_circ_sample/TISE_files/S.bin', 'r')
        #     S_test.load(S_viewer)
        #     S_viewer.destroy()
        #     S_test.assemble()

        # if os.path.exists('../../qprop_circ_sample/TISE_files/H.bin'):
        #     H_test = PETSc.Mat().createAIJ([self.input_params["splines"]["n_basis"]*self.input_params["lm"]["n_blocks"],self.input_params["splines"]["n_basis"]*self.input_params["lm"]["n_blocks"]],comm = PETSc.COMM_WORLD,nnz = 2*(self.input_params["splines"]["order"]-1)+1)
        #     H_viewer = PETSc.Viewer().createBinary('../../qprop_circ_sample/TISE_files/H.bin', 'r')
        #     H_test.load(H_viewer)
        #     H_viewer.destroy()
        #     H_test.assemble()

        # S_diff = S_total.copy()
        # S_diff.axpy(-1,S_test)
        # H_diff = H_total.copy()
        # H_diff.axpy(-1,H_test)

        # S_val = S_diff.norm(PETSc.NormType.FROBENIUS)
        # H_val = H_diff.norm(PETSc.NormType.FROBENIUS)

        # PETSc.Sys.Print(f"Norm of S_diff: {S_val}",comm=PETSc.COMM_WORLD)
        # PETSc.Sys.Print(f"Norm of H_diff: {H_val}",comm=PETSc.COMM_WORLD)
        ###############################################################




        

      
        S_L = S_total.copy()
        S_R = S_total.copy()
        
        self.S = S_total
    
        S_L.axpy(1j*self.input_params["box"]["time_spacing"]/2,H_total) 
        S_R.axpy(-1j*self.input_params["box"]["time_spacing"]/2,H_total)
        H_total.destroy()

        self.atomic_L = S_L
        self.atomic_R = S_R

        return None

    def computeNorm(self,state):
        S_norm = self.S.createVecRight()
        self.S.mult(state,S_norm)
        prod = state.dot(S_norm)
        S_norm.destroy()
        return np.real(prod)

    def computeNormIndices(self,Nt_total):
        norm_indices = np.linspace(0, Nt_total - 1, num=self.input_params["TDSE"]["norm_checkpoints"], dtype=int)
        return norm_indices

    def computeCheckpointIndices(self,Nt_total):
        checkpoint_indices = np.linspace(0, Nt_total - 1, num=self.input_params["TDSE"]["save_checkpoints"], dtype=int)
        return checkpoint_indices

    def loadStartingState(self):
        if not os.path.exists("TDSE_files/TDSE.h5"):
            return 0,self.state
        
        
        last_checkpoint = -1 
        if os.path.exists("TDSE_files/TDSE.h5"):
            with h5py.File("TDSE_files/TDSE.h5", "r") as f:
                # Look for datasets of the form psi_integer
                for key in f.keys():
                    if key.startswith("psi_"):
                        try:
                            psi_index = int(key.split("_")[1])
                            last_checkpoint= max(last_checkpoint, psi_index)
                        except ValueError:
                            continue
        
     
    
        checkpoint_state = PETSc.Vec().createMPI(self.input_params["splines"]["n_basis"]*self.input_params["lm"]["n_blocks"],comm = PETSc.COMM_WORLD)
        
        with h5py.File(f'TDSE_files/TDSE.h5', 'r') as f:
            data = f[f"/psi_{last_checkpoint}"][:]
            real_part = data[:,0]
            imaginary_part = data[:,1]
            total = real_part + 1j*imaginary_part

        global_indices = np.array(range(len(total)))
        global_indices = global_indices.astype("int32")
        checkpoint_state.setValues(global_indices,total)
        checkpoint_state.assemble()
        return last_checkpoint+1,checkpoint_state
        

        

    def propagateState(self,laserInstance):


        starting_idx,psi_initial = self.loadStartingState()
        current_norm = self.computeNorm(psi_initial)
        if rank == 0:
            norm_file = open("TDSE_files/norms.txt","w")
            norm_file.write(f"Norm of Initial state: {current_norm} \n")
            norm_file.close()

        if starting_idx == 0:
            # Starting from scratch
            ViewHDF5 = PETSc.Viewer().createHDF5("TDSE_files/TDSE.h5", mode=PETSc.Viewer.Mode.WRITE, comm=PETSc.COMM_WORLD)
        else:
            # Resuming from a checkpoint
            ViewHDF5 = PETSc.Viewer().createHDF5("TDSE_files/TDSE.h5", mode=PETSc.Viewer.Mode.APPEND, comm=PETSc.COMM_WORLD)

       
        norm_indices = self.computeNormIndices(self.input_params["box"]["Nt_total"] )
        checkpoint_indices = self.computeCheckpointIndices(self.input_params["box"]["Nt_total"])


        ksp = PETSc.KSP().create(comm = PETSc.COMM_WORLD)
        ksp.setTolerances(rtol = self.input_params["TDSE"]["tolerance"])

       
        Nt_total = self.input_params["box"]["Nt_total"]
        for idx in range(starting_idx,self.input_params["box"]["Nt_total"]):
            if rank == 0:
                print(f"Iteration {idx}/{Nt_total} \n")

            partial_L_copy = self.atomic_L.copy()
            partial_R_copy = self.atomic_R.copy()

            known = partial_R_copy.createVecRight() 
            solution = partial_L_copy.createVecRight()

            t = idx * self.input_params["box"]["time_spacing"]

            if idx < self.input_params["box"]["Nt"]:
                if self.input_params["laser"]["components"][0] or self.input_params["laser"]["components"][1]:
                    A_tilde = (laserInstance.Ax(t+self.input_params["box"]["time_spacing"]/2,self) + 1j*laserInstance.Ay(t+self.input_params["box"]["time_spacing"]/2,self))* 1j*self.input_params["box"]["time_spacing"]/2
                    A_tilde_star = (laserInstance.Ax(t+self.input_params["box"]["time_spacing"]/2,self) - 1j*laserInstance.Ay(t+self.input_params["box"]["time_spacing"]/2,self))* 1j*self.input_params["box"]["time_spacing"]/2
                    
                    partial_L_copy.axpy(A_tilde,self.hamiltonians[1],structure = petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
                    partial_R_copy.axpy(-A_tilde,self.hamiltonians[1],structure = petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

                    partial_L_copy.axpy(A_tilde_star,self.hamiltonians[0],structure = petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
                    partial_R_copy.axpy(-A_tilde_star,self.hamiltonians[0],structure = petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

                if self.input_params["laser"]["components"][2]:
                    Az = laserInstance.Az(t+self.input_params["box"]["time_spacing"]/2,self)*1j*self.input_params["box"]["time_spacing"]/2
                    partial_L_copy.axpy(Az,self.hamiltonians[2],structure = petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
                    partial_R_copy.axpy(-Az,self.hamiltonians[2],structure = petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

            partial_R_copy.mult(psi_initial,known)
            ksp.setOperators(partial_L_copy)
            ksp.solve(known,solution)
            solution.copy(psi_initial)

            Nt_total = self.input_params["box"]["Nt_total"]

            if idx in []:
                current_norm = self.computeNorm(psi_initial)
            if rank == 0 and idx in norm_indices:
                print(f"Norm of state at step {idx}/{Nt_total}: {current_norm} \n")
            
            if idx in checkpoint_indices:
                psi_initial.setName(f"psi_{idx}")
                ViewHDF5.view(psi_initial)

            partial_L_copy.destroy()
            partial_R_copy.destroy()
            known.destroy()
            solution.destroy()
            
        final_norm = self.computeNorm(psi_initial)

        if rank == 0:
            print(f"Norm of Final State:{final_norm}")

        ViewHDF5.destroy()
        ViewHDF5 = PETSc.Viewer().createHDF5("TDSE_files/TDSE.h5", mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)
        psi_initial.setName("psi_final")
        ViewHDF5.view(psi_initial)
        ViewHDF5.destroy()
        return None



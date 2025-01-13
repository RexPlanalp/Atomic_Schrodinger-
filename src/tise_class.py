import numpy as np
import json as json
import pprint as pprint

from collections.abc import Mapping
from collections import deque
from petsc4py import PETSc
from slepc4py import SLEPc
import petsc4py

import simulation_class as simulation

rank = PETSc.COMM_WORLD.rank
size = PETSc.COMM_WORLD.size

class tise(simulation.simulation):
    def __init__(self,input_file):
        super().__init__(input_file)
        

    def _solve_eigenvalue_problem(self,H,S,num_of_states):
        E = SLEPc.EPS().create(comm = PETSc.COMM_WORLD)
        E.setOperators(H,S)
        E.setDimensions(nev=num_of_states)
        E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        E.setTolerances(tol = self.input_params["TISE"]["tolerance"],max_it=self.input_params["TISE"]["max_iter"])
        E.solve()
        nconv = E.getConverged()
        return E,nconv
    
    def _create_K(self,basis):
        def _K_element(x,i,j,knots,order):
            return basis.dB(i, order, x, knots) * (1/2) * basis.dB(j, order, x, knots) 

        K = PETSc.Mat().createAIJ([self.input_params["splines"]["n_basis"],self.input_params["splines"]["n_basis"]],comm = PETSc.COMM_WORLD,nnz = 2*(self.input_params["splines"]["order"]-1) +1)
        istart,iend = K.getOwnershipRange()
        for i in range(istart,iend):
            j_start = max(0,i-self.input_params["splines"]["order"]+1)
            j_end = min(self.input_params["splines"]["n_basis"],i+self.input_params["splines"]["order"])
            for j in range(j_start,j_end):
                if abs(i-j)>= self.input_params["splines"]["order"]:
                    continue
                K_element = basis.integrate(_K_element,i,j,self.input_params["splines"]["order"],basis.knots)
                K.setValue(i,j,K_element)
        PETSc.COMM_WORLD.barrier()
        K.assemble() 
        return K 

    def _create_Inv_r2(self,basis):
        def _Inv_r2_element(x,i,j,knots,order):
            return basis.B(i, order, x, knots) * basis.B(j, order, x, knots) * 1/(x**2 + 1E-25)

        Inv_r2 = PETSc.Mat().createAIJ([self.input_params["splines"]["n_basis"],self.input_params["splines"]["n_basis"]],comm = PETSc.COMM_WORLD,nnz = 2*(self.input_params["splines"]["order"]-1) +1)
        istart,iend = Inv_r2.getOwnershipRange()
        for i in range(istart,iend):
            j_start = max(0,i-self.input_params["splines"]["order"]+1)
            j_end = min(self.input_params["splines"]["n_basis"],i+self.input_params["splines"]["order"])
            for j in range(j_start,j_end):
                if abs(i-j)>= self.input_params["splines"]["order"]:
                    continue
                Inv_r2_element = basis.integrate(_Inv_r2_element,i,j,self.input_params["splines"]["order"],basis.knots)
                Inv_r2.setValue(i,j,Inv_r2_element)
        PETSc.COMM_WORLD.barrier()
        Inv_r2.assemble() 
        return Inv_r2

    def _create_V(self,basis):
        def _V_element(x,i,j,knots,order):
            return basis.B(i, order, x, knots) * basis.B(j, order, x, knots) * self.input_params["box"]["potential_func"](x)

        V = PETSc.Mat().createAIJ([self.input_params["splines"]["n_basis"],self.input_params["splines"]["n_basis"]],comm = PETSc.COMM_WORLD,nnz = 2*(self.input_params["splines"]["order"]-1) +1)
        istart,iend = V.getOwnershipRange()
        for i in range(istart,iend):
            j_start = max(0,i-self.input_params["splines"]["order"]+1)
            j_end = min(self.input_params["splines"]["n_basis"],i+self.input_params["splines"]["order"])
            for j in range(j_start,j_end):
                if abs(i-j)>= self.input_params["splines"]["order"]:
                    continue
                V_element = basis.integrate(_V_element,i,j,self.input_params["splines"]["order"],basis.knots)
                V.setValue(i,j,V_element)
        PETSc.COMM_WORLD.barrier()
        V.assemble() 
        return V

    def _create_S(self,basis):
        def _S_element(x,i,j,knots,order):
            return basis.B(i, order, x, knots) * basis.B(j, order, x, knots)

        S = PETSc.Mat().createAIJ([self.input_params["splines"]["n_basis"],self.input_params["splines"]["n_basis"]],comm = PETSc.COMM_WORLD,nnz = 2*(self.input_params["splines"]["order"]-1) +1)
        istart,iend = S.getOwnershipRange()
        for i in range(istart,iend):
            j_start = max(0,i-self.input_params["splines"]["order"]+1)
            j_end = min(self.input_params["splines"]["n_basis"],i+self.input_params["splines"]["order"])
            for j in range(j_start,j_end):
                if abs(i-j)>= self.input_params["splines"]["order"]:
                    continue
                S_element = basis.integrate(_S_element,i,j,self.input_params["splines"]["order"],basis.knots)
                S.setValue(i,j,S_element)
        PETSc.COMM_WORLD.barrier()
        S.assemble() 
        return S

    def solve_eigensystem(self,basis):
        ViewTISE = PETSc.Viewer().createHDF5(f'TISE_files/{self.input_params["box"]["potential_func"].__name__}.h5', mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)
        
        basis.eta = 0
        K = self._create_K(basis)
        S = self._create_S(basis)
        V = self._create_V(basis)
        Inv_r2 = self._create_Inv_r2(basis)
        basis.eta = self.input_params["box"]["eta"]

        for l in range(self.input_params["lm"]["lmax"]+1):
            V_l = Inv_r2.copy()
            V_l.scale(l*(l+1)/2)
            V_l.axpy(1,V)

            H_l = K.copy()
            H_l.axpy(1,V_l)
        
            num_of_energies = self.input_params["lm"]["nmax"] - l 

            if num_of_energies > 0:
                petsc4py.PETSc.Sys.Print(f"Solving for l = {l}",comm=PETSc.COMM_WORLD)

                E,nconv = self._solve_eigenvalue_problem(H_l,S,num_of_energies)
                if rank == 0:
                    print(f"l = {l}, Requested:{num_of_energies}, Converged:{nconv}")
                for i in range(nconv):
                    eigenvalue = E.getEigenvalue(i) 
                    if eigenvalue.real > 0:
                        continue
                    eigen_vector = H_l.getVecLeft()  
                    E.getEigenvector(i, eigen_vector)  
                            
                    Sv = S.createVecRight()
                    S.mult(eigen_vector, Sv)
                    norm = eigen_vector.dot(Sv)

                    eigen_vector.scale(1/np.sqrt(norm))
                    eigen_vector.setName(f"Psi_{i+1+l}_{l}")
                    ViewTISE.view(eigen_vector)
                        
                    energy = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
                    energy.setValue(0,eigenvalue.real)
                    energy.setName(f"E_{i+1+l}_{l}")
                    energy.assemblyBegin()
                    energy.assemblyEnd()
                    ViewTISE.view(energy)
            H_l.destroy()
        ViewTISE.destroy()  
        K.destroy()
        S.destroy()
        V.destroy()
        Inv_r2.destroy()  
        return None
    
    def prepare_matrices(self,basis):
        K = self._create_K(basis)
        S = self._create_S(basis)
        V = self._create_V(basis)
        Inv_r2 = self._create_Inv_r2(basis)

        K_viewer = PETSc.Viewer(comm = PETSc.COMM_WORLD).createBinary("TISE_files/K.bin","w")
        K.view(K_viewer)
        K_viewer.destroy()

        S_viewer = PETSc.Viewer(comm = PETSc.COMM_WORLD).createBinary("TISE_files/S.bin","w")
        S.view(S_viewer)
        S_viewer.destroy()

        Invr2_viewer = PETSc.Viewer(comm = PETSc.COMM_WORLD).createBinary("TISE_files/Invr2.bin","w")
        Inv_r2.view(Invr2_viewer)
        Invr2_viewer.destroy()

        V_viewer = PETSc.Viewer(comm = PETSc.COMM_WORLD).createBinary("TISE_files/V.bin","w")
        V.view(V_viewer)
        V_viewer.destroy()
        return True
        

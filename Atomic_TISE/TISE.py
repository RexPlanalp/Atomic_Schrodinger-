import json 
import sys


from numpy import exp
from numpy import sqrt

sys.path.append('/users/becker/dopl4670/Research/Atomic_Schrodinger/Common')
import utility

from petsc4py import PETSc
from slepc4py import SLEPc

comm = PETSc.COMM_WORLD
rank = comm.rank

class Tise:
    def __init__(self, input_file):
        with open(input_file, 'r') as file:
            self.parameters = json.load(file)
        self.input_file = input_file

        if self.parameters["TISE"]["embed"]:
            from numpy.linalg import norm
            from numpy import cross


            polarization = self.parameters["lasers"]["polarization"]
            polarization /= norm(polarization)

            poynting = self.parameters["lasers"]["poynting"]
            poynting /= norm(poynting)

            ellipticity_Vector = cross(polarization, poynting) 
            ellipticity_Vector /= norm(ellipticity_Vector)

            ell = -self.parameters["lasers"]["ell"]
            
            components = [(1 if polarization[i] != 0 or ell * ellipticity_Vector[i] != 0 else 0) for i in range(3)]



            self.lm_dict,self.block_dict = utility.lm_expansion(self.parameters["lm"]["lmax"],self.parameters["lm"]["mmin"],self.parameters["lm"]["mmax"],self.parameters["lm"]["expansion"], \
                                                            self.parameters["state"], \
                                                                components)
            self.parameters["total_size"] = len(self.lm_dict) * self.parameters["splines"]["n_basis"]
            self.parameters["n_block"] = len(self.lm_dict)

            import matplotlib.pyplot as plt
            from numpy import zeros, flipud
            from os import mkdir,path

            if rank == 0:
                if not path.exists("images"):
                    mkdir("images")



            fig, ax = plt.subplots()
            grid_size = self.parameters["lm"]["lmax"] + 1
            grid = zeros((grid_size, 2 * self.parameters["lm"]["lmax"] + 1))

            for l, m in self.lm_dict.keys():
                grid[self.parameters["lm"]["lmax"] - l, m + self.parameters["lm"]["lmax"]] = 1

            ax.imshow(flipud(grid), cmap='gray', interpolation='none', origin='lower')
            ax.set_xlabel('m')
            ax.set_ylabel('l')
            ax.set_xticks([i for i in range(0, 2 * self.parameters["lm"]["lmax"] + 1, 10)])  # Positions for ticks
            ax.set_xticklabels([str(i - self.parameters["lm"]["lmax"]) for i in range(0, 2 * self.parameters["lm"]["lmax"] + 1, 10)])  # Labels from -lmax to lmax
            ax.set_title('Reachable (white) and Unreachable (black) Points in l-m Space')
            if rank == 0:
                plt.savefig("images/lm_space.png")

        def H(x):
            return (-1/(x+1E-25))          
        def He(x):
                return (-1/sqrt(x**2 + 1E-25)) + -1.0*exp(-2.0329*x)/sqrt(x**2 + 1E-25)  - 0.3953*exp(-6.1805*x)
        def Ar(x):
            return -1.0 / (x + 1e-25) - 17.0 * exp(-0.8103 * x) / (x + 1e-25) \
            - (-15.9583) * exp(-1.2305 * x) \
            - (-27.7467) * exp(-4.3946 * x) \
            - 2.1768 * exp(-86.7179 * x)
        
        if self.parameters["species"] == "H":
            self.pot_func = H
        elif self.parameters["species"] == "He":
            self.pot_func = He
        elif self.parameters["species"] == "Ar":
            self.pot_func = Ar
        
    def get(self, key, default=None):
        return self.parameters.get(key, default)
    
    def __getitem__(self, key):
        return self.parameters[key]
    
    def __setitem__(self, key, value):
        self.parameters[key] = value
    
    def __repr__(self):
        return f"TISE({self.input_file})"
    
    def _solveEigenvalueProblem(self,H,S,num_of_states):
        E = SLEPc.EPS().create(comm = PETSc.COMM_WORLD)
        E.setOperators(H,S)
        E.setDimensions(nev=num_of_states)
        E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        E.setTolerances(tol = self.parameters["TISE"]["tolerance"],max_it=self.parameters["TISE"]["max_iter"])
        E.solve()
        nconv = E.getConverged()
        return E,nconv
    
    def _createK(self,basisInstance):
        n_basis = self.parameters["splines"]["n_basis"]
        order = self.parameters["splines"]["order"]

        knots = basisInstance.knots
        dB = basisInstance.dB
       
        def _K_element(x,i,j,knots,order):
            return dB(i, order, x, knots) * (1/2) * dB(j, order, x, knots) 

        K = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD,nnz = 2*(order-1) +1)
        istart,iend = K.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                if abs(i-j)>= order:
                    continue
                K_element = basisInstance.integrate(_K_element,i,j,order,knots)
                K.setValue(i,j,K_element)
        comm.barrier()
        K.assemble() 
        return K 

    def _createV_l(self,basisInstance,l):
        n_basis = self.parameters["splines"]["n_basis"]
        order = self.parameters["splines"]["order"]

        knots = basisInstance.knots
        B = basisInstance.B
       
        def _V_element(x,i,j,knots,order):
            return (B(i, order, x, knots) * B(j, order,x, knots) * l*(l+1)/(2*x**2 + 1E-25) + B(i, order, x, knots) * B(j, order, x, knots)* self.pot_func(x))

        V = PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*(order-1) +1)
        istart,iend = V.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                if abs(i-j)>=order:
                    continue
                V_element = basisInstance.integrate(_V_element,i,j,order,knots)
                V.setValue(i,j,V_element)
        comm.barrier()
        V.assemble() 
        return V   

    def _createS(self,basisInstance):
        n_basis = self.parameters["splines"]["n_basis"]
        order = self.parameters["splines"]["order"]

        knots = basisInstance.knots
        B = basisInstance.B

        def _S_element(x,i,j,knots,order):
            return (B(i, order, x, knots) * B(j, order,x,knots))

        S = PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*(order-1) +1)
        istart,iend = S.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                    if abs(i-j)>=order:
                        continue
                    S_element = basisInstance.integrate(_S_element,i,j,order,knots)
                    S.setValue(i,j,S_element)
        comm.barrier()    
        S.assemble()
        if self.parameters["TISE"]["embed"]:
            self._embedS(S)
        return S
        
    def _embedS(self,S):
        n_block = self.parameters["n_block"]
        n_basis = self.parameters["splines"]["n_basis"]
        order = self.parameters["splines"]["order"]

        kron = utility.kron

        I = PETSc.Mat().createAIJ([n_block,n_block],comm = PETSc.COMM_WORLD)
        istart,iend = I.getOwnershipRange()
        for i in range(istart,iend):
            I.setValue(i,i,1)
        comm.barrier()
        I.assemble()

        S_atom = kron(I,S,comm,2*(order-1)+1)
        S_viewer = PETSc.Viewer().createBinary("TISE_files/S.bin","w")
        S_atom.view(S_viewer)
        S_viewer.destroy()
        I.destroy()

        return None

    def solveEigensystem(self,basisInstance):
        n_block = self.parameters["n_block"]

        n_basis = self.parameters["splines"]["n_basis"]
        order = self.parameters["splines"]["order"]

        lmax = self.parameters["lm"]["lmax"]
        nmax = self.parameters["lm"]["nmax"]

        kron = utility.kron

        if self.parameters["TISE"]["embed"]:
            H_atom = PETSc.Mat().createAIJ([n_block*n_basis,n_block*n_basis],comm = comm,nnz = 2*(order-1)+1)
            H_atom.assemble()


        ViewTISE = PETSc.Viewer().createHDF5(f"{self.pot_func.__name__}.h5", mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)
        
        K = self._createK(basisInstance)
        S = self._createS(basisInstance)

        for l in range(lmax+1):
            V_l = self._createV_l(basisInstance,l)
            H_l = K + V_l
            
            block_indices = []
            for (lprime,mprime),block_idx in self.lm_dict.items():
                if lprime == l:
                    block_indices.append(block_idx)
                
            
            partial_I = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 1)
            istart,iend = partial_I.getOwnershipRange()
            for i in range(istart,iend):
                if i in block_indices:
                    partial_I.setValue(i,i,1)
            comm.barrier()
            partial_I.assemble()

            partial_H = kron(partial_I,H_l,comm,2*(order-1) +1)
            H_atom.axpy(1,partial_H)
            partial_I.destroy()


            num_of_energies = nmax - l 

            if rank == 0:
                if num_of_energies > 0 and self.parameters["TISE"]["embed"]:
                    print(f"Solving and Embedding for l = {l}")
                elif self.parameters["TISE"]["embed"]:
                    print(f"Embedding for l = {l}")

            if num_of_energies > 0:

                E,nconv = self._solveEigenvalueProblem(H_l,S,num_of_energies)
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

                    eigen_vector.scale(1/sqrt(norm))
                    eigen_vector.setName(f"Psi_{i+1+l}_{l}")
                    ViewTISE.view(eigen_vector)
                        
                    energy = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
                    energy.setValue(0,eigenvalue.real)
                    energy.setName(f"E_{i+1+l}_{l}")
                    energy.assemblyBegin()
                    energy.assemblyEnd()
                    ViewTISE.view(energy)
            H_l.destroy()
        

        if self.parameters["TISE"]["embed"]:
            H_viewer = PETSc.Viewer().createBinary("TISE_files/H.bin","w")
            H_atom.view(H_viewer)
            H_viewer.destroy()
        ViewTISE.destroy()    
        return None
    

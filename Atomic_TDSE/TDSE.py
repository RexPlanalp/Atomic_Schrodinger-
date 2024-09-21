import json 
import sys
import h5py
import numpy as np
import os

sys.path.append('/users/becker/dopl4670/Research/Atomic_Schrodinger/Common')
import utility

from petsc4py import PETSc
from slepc4py import SLEPc
import petsc4py

comm = PETSc.COMM_WORLD
rank = comm.rank

class Tdse:
    def __init__(self, input_file):
        with open(input_file, 'r') as file:
            self.parameters = json.load(file)
        self.input_file = input_file

        
        self.lm_dict,self.block_dict = utility.lm_expansion(self.parameters["lm"]["lmax"], \
                                                            self.parameters["state"], \
                                                                self.parameters["lasers"]["polarization"])
        self.parameters["total_size"] = len(self.lm_dict) * self.parameters["splines"]["n_basis"]
        self.parameters["n_block"] = len(self.lm_dict)

        if rank == 0:
            print(f"Total Dimensionality: {self.parameters['total_size']} X  {self.parameters['total_size']}")
            print("\n")

    def get(self, key, default=None):
        return self.parameters.get(key, default)
    
    def __getitem__(self, key):
        return self.parameters[key]
    
    def __setitem__(self, key, value):
        self.parameters[key] = value
    
    def __repr__(self):
        return f"TDSE({self.input_file})"
    
    def readInState(self):
        n_basis = self.parameters["splines"]["n_basis"]
        n_block = self.parameters["n_block"]
        potential = self.parameters["species"]
        n_value, l_value, m_value = self.parameters["state"]

        block_index = self.lm_dict[(l_value, m_value)]
        psi_initial = PETSc.Vec().createMPI(self.parameters["total_size"],comm = PETSc.COMM_WORLD)
        
        with h5py.File(f'TISE_files/{potential}.h5', 'r') as f:
            data = f[f"/Psi_{n_value}_{l_value}"][:]
            real_part = data[:,0]
            imaginary_part = data[:,1]
            total = real_part + 1j*imaginary_part

        global_indices = np.array(range(n_basis))+block_index*n_basis
        global_indices = global_indices.astype("int32")
        psi_initial.setValues(global_indices,total)
        psi_initial.assemble()
        self.state = psi_initial
        return None

    def constructInteraction(self,basisInstance):
        self.hamiltonians = [0]*3

        n_block = self.parameters["n_block"]
        n_basis = self.parameters["splines"]["n_basis"]
        order = self.parameters["splines"]["order"]
        polarization = self.parameters["lasers"]["polarization"]
        kron = utility.kron

        knots = basisInstance.knots
        B = basisInstance.B
        dB = basisInstance.dB
        integrate = basisInstance.integrate

        def _H_der_R_element(x,i,j,knots,order):
            return (B(i, order, x, knots)*dB(j, order, x, knots))
        H_der_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD,nnz = 2*(order-1)+1)
        istart,iend = H_der_R.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                if np.abs(i-j)>=order:
                    continue
                H_element = integrate(_H_der_R_element,i,j,order,knots)
                H_der_R.setValue(i,j,H_element)
        PETSc.COMM_WORLD.barrier()
        H_der_R.assemble()

        def _H_inv_R_element(x,i,j,knots,order):
            return (B(i, order, x, knots)*B(j, order, x, knots)/ (x+1E-25))
        H_inv_R =  PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD,nnz = 2*(order-1)+1)
        istart,iend = H_inv_R.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                if np.abs(i-j)>=order:
                    continue
                H_element = integrate(_H_inv_R_element,i,j,order,knots)
                H_inv_R.setValue(i,j,H_element)
        PETSc.COMM_WORLD.barrier()
        H_inv_R.assemble()


        if polarization[0] or polarization[1]:
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

            H_XY_lm_1 = PETSc.Mat().createAIJ([n_block,n_block],comm = PETSc.COMM_WORLD,nnz = 2)
            istart,iend = H_XY_lm_1.getOwnershipRange()
            for i in range(istart,iend):
                l,m = self.block_dict[i]
                for j in range(n_block):
                    lprime,mprime = self.block_dict[j]
                    if (l == lprime+1) and (m == mprime+1):
                        H_XY_lm_1.setValue(i,j,a(l,m))
                    elif (l == lprime-1) and (m == mprime+1):
                        H_XY_lm_1.setValue(i,j,b(l,m))
            PETSc.COMM_WORLD.barrier()
            H_XY_lm_1.assemble()

            H_XY_lm_2 = PETSc.Mat().createAIJ([n_block,n_block],comm = PETSc.COMM_WORLD,nnz = 2)
            istart,iend = H_XY_lm_2.getOwnershipRange()
            for i in range(istart,iend):
                l,m = self.block_dict[i]
                for j in range(n_block):
                    lprime,mprime = self.block_dict[j]
                    if (l == lprime+1) and (m == mprime+1):
                        H_XY_lm_2.setValue(i,j,c(l,m))
                    elif (l == lprime-1) and (m == mprime+1):
                        H_XY_lm_2.setValue(i,j,-d(l,m))
            PETSc.COMM_WORLD.barrier()
            H_XY_lm_2.assemble()


            term1 = kron(H_XY_lm_1,H_inv_R,comm,2*(2*(order-1) + 1))
            term2 = kron(H_XY_lm_2,H_der_R,comm,2*(2*(order-1) + 1))

            term1.axpy(1,term2)
            term1.scale(1j/2)
            term2.destroy()
            H_XY_lm_1.destroy()
            H_XY_lm_2.destroy()

            self.hamiltonians[0] = term1


            H_XY_lm_3 = PETSc.Mat().createAIJ([n_block,n_block],comm = PETSc.COMM_WORLD,nnz = 2)
            istart,iend = H_XY_lm_3.getOwnershipRange()
            for i in range(istart,iend):
                l,m = self.block_dict[i]
                for j in range(n_block):
                    lprime,mprime = self.block_dict[j]
                    if (l == lprime+1) and (m == mprime-1):
                        H_XY_lm_3.setValue(i,j,atilde(l,m))
                    elif (l == lprime-1) and (m == mprime-1):
                        H_XY_lm_3.setValue(i,j,btilde(l,m))
            PETSc.COMM_WORLD.barrier()
            H_XY_lm_3.assemble()

            H_XY_lm_4 = PETSc.Mat().createAIJ([n_block,n_block],comm = PETSc.COMM_WORLD,nnz = 2)
            istart,iend = H_XY_lm_4.getOwnershipRange()
            for i in range(istart,iend):
                l,m = self.block_dict[i]
                for j in range(n_block):
                    lprime,mprime = self.block_dict[j]
                    if (l == lprime+1) and (m == mprime-1):
                        H_XY_lm_4.setValue(i,j,-ctilde(l,m))
                    elif (l == lprime-1) and (m == mprime-1):
                        H_XY_lm_4.setValue(i,j,dtilde(l,m))
            H_XY_lm_4.assemble()

            term3 = kron(H_XY_lm_3,H_inv_R,comm,2*(2*(order-1) + 1))
            term4 = kron(H_XY_lm_4,H_der_R,comm,2*(2*(order-1) + 1))

            term3.axpy(1,term4)
            term3.scale(1j/2)
            term4.destroy()
            H_XY_lm_3.destroy()
            H_XY_lm_4.destroy()
            
            self.hamiltonians[1] = term3
            
        if polarization[2]:
            def clm(l,m):
                return -1j*np.sqrt(((l+1)**2 - m**2)/((2*l+1)*(2*l+3)))
            def dlm(l,m):
                return -1j*np.sqrt(((l)**2 - m**2)/((2*l-1)*(2*l+1)))
            
            H_Z_lm_1 = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 2)
            istart,iend = H_Z_lm_1.getOwnershipRange()
            for i in range(istart,iend):
                l,m = self.block_dict[i]
                for j in range(n_block):
                    lprime,mprime = self.block_dict[j]
                    if l == lprime+1:
                        H_Z_lm_1.setValue(i,j,dlm(l,m))
                    elif l == lprime-1:
                        H_Z_lm_1.setValue(i,j,clm(l,m))
            PETSc.COMM_WORLD.barrier()
            H_Z_lm_1.assemble()

            H_Z_lm_2 = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 2)
            istart,iend = H_Z_lm_2.getOwnershipRange()
            for i in range(istart,iend):
                l,m = self.block_dict[i]
                for j in range(n_block):
                    lprime,mprime = self.block_dict[j]
                    if l == lprime+1:
                        H_Z_lm_2.setValue(i,j,-(l)*dlm(l,m))
                    elif l == lprime-1:
                        H_Z_lm_2.setValue(i,j,(l+1)*clm(l,m))
            PETSc.COMM_WORLD.barrier()
            H_Z_lm_2.assemble()

            H_Z_1 = kron(H_Z_lm_1,H_der_R,comm,2*(2*(order-1)+1))
            H_Z_2 = kron(H_Z_lm_2,H_inv_R,comm,2*(2*(order-1)+1))
            H_Z_lm_1.destroy()
            H_Z_lm_2.destroy()

            H_Z_1.axpy(1.0,H_Z_2)

            self.hamiltonians[2] = H_Z_1

    def constructHHG(self,basisInstance):
        if not self.parameters["HHG"]:
            return None
        self.hhgs = [0]*3

        n_block = self.parameters["n_block"]
        n_basis = self.parameters["splines"]["n_basis"]
        order = self.parameters["splines"]["order"]
        polarization = self.parameters["lasers"]["polarization"]
        kron = utility.kron

        knots = basisInstance.knots
        B = basisInstance.B
        integrate = basisInstance.integrate

        def _H_inv_2_R_element(x,i,j,knots,order):
            return (B(i, order, x, knots)*B(j, order, x, knots)/(x**2+1E-25))
        H_inv_2_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD,nnz = 2*(order-1)+1)
        istart,iend = H_inv_2_R.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                if np.abs(i-j)>=order:
                    continue
                H_element = integrate(_H_inv_2_R_element,i,j,order,knots)
                H_inv_2_R.setValue(i,j,H_element)
        PETSc.COMM_WORLD.barrier()
        H_inv_2_R.assemble()

        def alpha(l,m):
            f1 = np.sqrt(((l+m-1)*(l+m))/(2*(2*l+1)*(2*l-1)))
            return f1/np.sqrt(2)
        def beta(l,m):
            f1 = np.sqrt(((l-m+1)*(l-m+2)*(l+1))/((2*l +1)*(2*l + 2)*(2*l+3)))
            return -f1/np.sqrt(2)
        
        def charlie(l,m):
            f1 = np.sqrt(((l-m-1)*(l-m))/(2*(2*l+1)*(2*l-1)))
            return f1/np.sqrt(2)
        def delta(l,m):
            f1 = np.sqrt(((l+m+1)*(l+m+2)*(l+1))/((2*l +1)*(2*l + 2)*(2*l+3)))
            return -f1/np.sqrt(2)
        
        def echo(l,m):
            f1 = np.sqrt(((l+m)*(l-m))/((2*l -1)*(2*l+1)))
            return f1

        def foxtrot(l,m):
            f1 = np.sqrt(((l+m+1)*(l-m+1))/((2*l +1)*(2*l + 3)))
            return f1
        
        if polarization[0]:
            HHG_x_lm = PETSc.Mat().createAIJ([n_block,n_block],comm = PETSc.COMM_WORLD,nnz = 4)
            istart,iend = HHG_x_lm.getOwnershipRange()
            for i in range(istart,iend):
                l,m = self.block_dict[i]
                for j in range(n_block):
                    lprime,mprime = self.block_dict[j]

                    # Corresponds to <1,-1>
                    if (l == lprime+1) and (m == mprime-1):
                        HHG_x_lm.setValue(i,j,charlie(l,m))
                    elif (l == lprime-1) and (m == mprime-1):
                        HHG_x_lm.setValue(i,j,delta(l,m))

                    # Corresponds to -<1,1>
                    elif (l == lprime+1) and (m == mprime+1):
                        HHG_x_lm.setValue(i,j,-alpha(l,m))
                    elif (l == lprime-1) and (m == mprime+1):
                        HHG_x_lm.setValue(i,j,-beta(l,m))
            PETSc.COMM_WORLD.barrier()
            HHG_x_lm.assemble()

            HHG_x = kron(HHG_x_lm,H_inv_2_R,comm,4*(2*(order-1) + 1))
            self.hhgs[0] = HHG_x

        if polarization[1]:
            HHG_y_lm = PETSc.Mat().createAIJ([n_block,n_block],comm = PETSc.COMM_WORLD,nnz = 4)
            istart,iend = HHG_y_lm.getOwnershipRange()
            for i in range(istart,iend):
                l,m = self.block_dict[i]
                for j in range(n_block):
                    lprime,mprime = self.block_dict[j]

                    # Corresponds to <1,-1>
                    if (l == lprime+1) and (m == mprime-1):
                        HHG_y_lm.setValue(i,j,charlie(l,m))
                    elif (l == lprime-1) and (m == mprime-1):
                        HHG_y_lm.setValue(i,j,delta(l,m))

                    # Corresponds to <1,1>
                    elif (l == lprime+1) and (m == mprime+1):
                        HHG_y_lm.setValue(i,j,alpha(l,m))
                    elif (l == lprime-1) and (m == mprime+1):
                        HHG_y_lm.setValue(i,j,beta(l,m))
            PETSc.COMM_WORLD.barrier()
            HHG_y_lm.assemble()

            HHG_y = kron(HHG_y_lm,H_inv_2_R,comm,4*(2*(order-1) + 1))
            self.hhgs[1] = HHG_y
                    
        if polarization[2]:
            HHG_z_lm = PETSc.Mat().createAIJ([n_block,n_block],comm = PETSc.COMM_WORLD,nnz = 2)
            istart,iend = HHG_z_lm.getOwnershipRange()
            for i in range(istart,iend):
                l,m = self.block_dict[i]
                for j in range(n_block):
                    lprime,mprime = self.block_dict[j]

                    if (l == lprime+1) and (m == mprime):
                        HHG_z_lm.setValue(i,j,echo(l,m))
                    elif (l == lprime-1) and (m == mprime):
                        HHG_z_lm.setValue(i,j,foxtrot(l,m))

                    
            PETSc.COMM_WORLD.barrier()
            HHG_z_lm.assemble()

            HHG_z = kron(HHG_z_lm,H_inv_2_R,comm,2*(2*(order-1) + 1))
            self.hhgs[2] = HHG_z

    def constructAtomic(self):
        order = self.parameters["splines"]["order"]

        total_size = self.parameters["total_size"]

        dt = self.parameters["box"]["time_spacing"]

        if os.path.exists('TISE_files/S.bin'):
            S = PETSc.Mat().createAIJ([total_size,total_size],comm = PETSc.COMM_WORLD,nnz = 2*(order-1)+1)
            S_viewer = PETSc.Viewer().createBinary('TISE_files/S.bin', 'r')
            S.load(S_viewer)
            S_viewer.destroy()
            S.assemble()
            

        if os.path.exists('TISE_files/H.bin'):
            H = PETSc.Mat().createAIJ([total_size,total_size],comm = PETSc.COMM_WORLD,nnz = 2*(order-1)+1)
            H_viewer = PETSc.Viewer().createBinary('TISE_files/H.bin', 'r')
            H.load(H_viewer)
            H_viewer.destroy()
            H.assemble()
            
        
        S_L = S.copy()
        S_R = S.copy()
        
        self.S = S
    
        S_L.axpy(1j*dt/2,H)
        S_R.axpy(-1j*dt/2,H)
        H.destroy()

        self.atomic_L = S_L
        self.atomic_R = S_R

        return None

    def propagateState(self,laserInstance):
        dt = self.parameters["box"]["time_spacing"]
        polarization = self.parameters["lasers"]["polarization"]

        psi_initial = self.state

        S_norm = self.S.createVecRight()
        self.S.mult(psi_initial,S_norm)
        prod = psi_initial.dot(S_norm)
        S_norm.destroy()
        
        if rank == 0:
            norm_file = open("TDSE_files/norms.txt","w")
            norm_file.write(f"Norm of Initial state: {np.real(prod)} \n")
            norm_file.close()

        total_iterations = len(laserInstance.total_time)
        norm_indices = np.linspace(0, total_iterations - 1, num=100, dtype=int)


        ksp = PETSc.KSP().create(comm = PETSc.COMM_WORLD)
        ksp.setTolerances(rtol = self.parameters["TDSE"]["tolerance"])

        Nt = len(laserInstance.t)
        Nt_post = len(laserInstance.t_post)

        if self.parameters["HHG"]:
            HHG_data = np.zeros((3,Nt+Nt_post),dtype = np.complex128)
            laser_data = np.zeros((3,Nt+Nt_post),dtype = np.complex128)

        for idx in range(total_iterations):

            if self.parameters["HHG"]:
                if polarization[0]:
                    temp = self.hhgs[0].createVecRight()
                    self.hhgs[0].mult(psi_initial, temp)
                    prodx = psi_initial.dot(temp)
                    temp.destroy()
                    HHG_data[0,idx] = prodx
                    laser_data[0,idx] = laserInstance.Ax_func(idx*dt)
                if polarization[1]:
                    temp = self.hhgs[1].createVecRight()
                    self.hhgs[1].mult(psi_initial, temp)
                    prody = psi_initial.dot(temp)
                    temp.destroy()
                    HHG_data[1,idx] = prody
                    laser_data[1,idx] = laserInstance.Ay_func(idx*dt)
                if polarization[2]:
                    temp = self.hhgs[2].createVecRight()
                    self.hhgs[2].mult(psi_initial, temp)
                    prodz = psi_initial.dot(temp)
                    temp.destroy()
                    HHG_data[2,idx] = prodz
                    laser_data[2,idx] = laserInstance.Az_func(idx*dt)




            partial_L_copy = self.atomic_L.copy()
            partial_R_copy = self.atomic_R.copy()

            t = idx * dt
            if PETSc.COMM_WORLD.rank == 0:
                    print(idx,total_iterations)
            if idx < Nt:
                if polarization[0] or polarization[1]:
                    A_tilde = (laserInstance.Ax_func(t+dt/2) + 1j*laserInstance.Ay_func(t+dt/2))* 1j*dt/2
                    A_tilde_star = (laserInstance.Ax_func(t+dt/2) - 1j*laserInstance.Ay_func(t+dt/2))* 1j*dt/2
                    
                    partial_L_copy.axpy(A_tilde,self.hamiltonians[1],structure = petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
                    partial_R_copy.axpy(-A_tilde,self.hamiltonians[1],structure = petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

                    partial_L_copy.axpy(A_tilde_star,self.hamiltonians[0],structure = petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
                    partial_R_copy.axpy(-A_tilde_star,self.hamiltonians[0],structure = petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

                if polarization[2]:
                    Az = laserInstance.Az_func(t+dt/2)*1j*dt/2
                    partial_L_copy.axpy(Az,self.hamiltonians[2],structure = petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
                    partial_R_copy.axpy(-Az,self.hamiltonians[2],structure = petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)


                known = partial_R_copy.createVecRight() 
                solution = partial_L_copy.createVecRight()
                partial_R_copy.mult(psi_initial,known)

                ksp.setOperators(partial_L_copy)
                ksp.solve(known,solution)
                solution.copy(psi_initial)

                S_norm = self.S.createVecRight()
                self.S.mult(psi_initial,S_norm)
                prod = psi_initial.dot(S_norm)
                S_norm.destroy()

                if rank == 0 and idx in norm_indices:
                    norm_file = open("TDSE_files/norms.txt","a")
                    norm_file.write(f"Norm of state at step {idx}: {np.real(prod)} \n")
                    norm_file.close()

                partial_L_copy.destroy()
                partial_R_copy.destroy()
                known.destroy()
                solution.destroy()
            else:
                ksp.setOperators(self.atomic_L)

                known = self.atomic_R.createVecRight()
                solution = partial_L_copy.createVecRight()
                self.atomic_R.mult(psi_initial, known)

                ksp.solve(known, solution)
                solution.copy(psi_initial)

                S_norm = self.S.createVecRight()
                self.S.mult(psi_initial, S_norm)
                prod = psi_initial.dot(S_norm)
                S_norm.destroy()

                if rank == 0 and idx in norm_indices:
                    norm_file = open("TDSE_files/norms.txt","a")
                    norm_file.write(f"Norm of state at step {idx}: {np.real(prod)} \n")
                    norm_file.close()

                partial_L_copy.destroy()
                partial_R_copy.destroy()
                known.destroy()
                solution.destroy()

        S_norm = self.S.createVecRight()
        self.S.mult(psi_initial,S_norm)
        prod = psi_initial.dot(S_norm)
        S_norm.destroy()
        self.S.destroy()

        if rank == 0:
            print(f"Norm of Final State:{np.real(prod)}")

        if self.parameters["HHG"]:
            np.save("TDSE_files/HHG_data.npy",HHG_data)
            np.save("TDSE_files/laser_data.npy",laser_data)
            
        ViewHDF5 = PETSc.Viewer().createHDF5("TDSE.h5", mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)
        psi_initial.setName("psi_final")
        ViewHDF5.view(psi_initial)
        ViewHDF5.destroy()

        return None



import sys as sys

import tise_class as tise
import tdse_class as tdse
import basis_class as basis
# import PES as PES

import laser_class as laser
import os as os
import time as time

input_file = "input.json"

from petsc4py import PETSc
from slepc4py import SLEPc
import petsc4py

rank = PETSc.COMM_WORLD.rank
size = PETSc.COMM_WORLD.size

if "TISE" in sys.argv:
    total_start = time.time()

    petsc4py.PETSc.Sys.Print("Creating Directories")
    start = time.time()
    if rank == 0:
        if not os.path.exists("TISE_files"):
            os.system("mkdir TISE_files")
        if not os.path.exists("images"):
            os.system("mkdir images")
        if not os.path.exists("temp"):
            os.system("mkdir temp")
    end = time.time()
    petsc4py.PETSc.Sys.Print("Time to create directories: ", end-start)

    petsc4py.PETSc.Sys.Print("Initializing Simulation")
    start = time.time()
    sim = tise.tise(input_file)
    end = time.time()
    petsc4py.PETSc.Sys.Print("Time to initialize simulation: ", end-start)

    petsc4py.PETSc.Sys.Print("Setting up bsplines")
    start = time.time()
    bspline = basis.basis(sim)
    end = time.time()
    petsc4py.PETSc.Sys.Print("Time to set up bsplines: ", end-start)

    petsc4py.PETSc.Sys.Print("Solving TISE:")
    start = time.time()
    sim.solve_eigensystem(bspline)
    end = time.time()
    petsc4py.PETSc.Sys.Print("Time to solve TISE: ", end-start)

    petsc4py.PETSc.Sys.Print("Preparing Matrices")
    start = time.time()
    sim.prepare_matrices(bspline)
    end = time.time()
    petsc4py.PETSc.Sys.Print("Time to prepare matrices: ", end-start)

    total_end = time.time()
    petsc4py.PETSc.Sys.Print("Total time: ", total_end-total_start)
    petsc4py.PETSc.Sys.Print("Cleaning up...")
    if rank == 0:
        os.system("rm -rf temp")

PETSc.COMM_WORLD.barrier()

if "TDSE" in sys.argv:
    if rank == 0:
        if not os.path.exists("TDSE_files"):
            os.system("mkdir TDSE_files")
        if not os.path.exists("temp"):
            os.system("mkdir temp")
    sim = tdse.tdse(input_file)
    field = laser.laser(sim)
    field.plotPulse(sim)
    bspline = basis.basis(sim)
    sim.readInState()
    sim.constructInteraction(bspline)
    sim.constructAtomic()
    sim.propagateState(field)
    if rank == 0:
        os.system("rm -rf temp")
   

# elif sys.argv[1] == "PES":
#     if rank == 0:
#         os.mkdir("PES_files")
   
#     sim = PES.PES(input_file)
#     sim.setup_simulation()
#     basis = Basis.basis(sim)
#     sim.loadS()
#     sim.load_final_state()
#     sim.project_out_bound_states()
#     sim.expand_final_state(basis)
#     sim.compute_partial_spectra()
#     sim.compute_angle_integrated()
#     sim.compute_angle_resolved()
    


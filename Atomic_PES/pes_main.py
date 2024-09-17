import os
import time
import sys

from petsc4py import PETSc
rank = PETSc.COMM_WORLD.rank

from PES import *
sys.path.append('/users/becker/dopl4670/Research/Atomic_Schrodinger/Common')
from Basis import basis

if rank == 0:
    if not os.path.exists("PES_files"):
        os.mkdir("PES_files")

if rank == 0:
    init_start = time.time()
    print("Initializing Simulation")
    print("\n")
input_file = "input.json"
PESInstance = PES(input_file)
PESInstance.loadS_R()
PESInstance.load_final_state()
basisInstance = basis(PESInstance)
if rank == 0:
    init_end = time.time()
    print(f"Initializing Simulation Finished in {round(init_end - init_start, 2)} seconds")
    print("\n")


if rank == 0:
    init_start = time.time()
    print("Projecting Out Bound States")
    print("\n")
PESInstance.project_out_bound_states()
if rank == 0:
    init_end = time.time()
    print(f"Projecting Out Bound States Finished in {round(init_end - init_start, 2)} seconds")
    print("\n")

if rank == 0:
    init_start = time.time()
    print("Expanding Wavefunction In Position Space")
    print("\n")
PESInstance.expand_final_state(basisInstance)
if rank == 0:
    init_end = time.time()
    print(f"Expanding Wavefunction In Position Space Finished in {round(init_end - init_start, 2)} seconds")
    print("\n")

if rank == 0:
    init_start = time.time()
    print("Computing Continuum States")
    print("\n")
PESInstance.compute_continuum_states()
if rank == 0:
    init_end = time.time()
    print(f"Computing Continuum States Finished in {round(init_end - init_start, 2)} seconds")
    print("\n")

if rank == 0:
    init_start = time.time()
    print("Computing Partial Spectra")
    print("\n")
PESInstance.compute_partial_spectra()
if rank == 0:
    init_end = time.time()
    print(f"Computing Partial Spectra Finished in {round(init_end - init_start, 2)} seconds")
    print("\n")

if rank == 0:
    init_start = time.time()
    print("Computing Photoelectron Spectra")
    print("\n")
PESInstance.compute_angle_integrated()
PESInstance.compute_angle_resolved()
if rank == 0:
    init_end = time.time()
    print(f"Computing Photoelectron Spectra Finished in {round(init_end - init_start, 2)} seconds")
    print("\n")







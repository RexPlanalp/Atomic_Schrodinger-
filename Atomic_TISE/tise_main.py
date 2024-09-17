import os
import time
import sys

from petsc4py import PETSc
rank = PETSc.COMM_WORLD.rank

from TISE import Tise
sys.path.append('/users/becker/dopl4670/Research/Atomic_Schrodinger/Common')
from Basis import basis

if rank == 0:
    if not os.path.exists("TISE_files"):
        os.mkdir("TISE_files")
    if not os.path.exists("temp"):
        os.mkdir("temp")

if rank == 0:
    init_start = time.time()
    print("Initializing Simulation")
    print("\n")
input_file = "input.json"
TISEInstance = Tise(input_file)
basisInstance = basis(TISEInstance)
if rank == 0:
    init_end = time.time()
    print(f"Initializing Simulation Finished in {round(init_end - init_start, 2)} seconds")
    print("\n")

if rank == 0:
    sim_start = time.time()
    print("Solving TISE:")
    print("\n")
TISEInstance.solveEigensystem(basisInstance)
if rank == 0:
    sim_end = time.time()
    print(f"TISE Solver Finished in {round(sim_end - sim_start, 2)} seconds")
    print("\n")

if rank == 0:
    print("Cleaning up...")
    os.system("rm -rf temp")
    os.system(f"mv {TISEInstance.pot_func.__name__}.h5 TISE_files")









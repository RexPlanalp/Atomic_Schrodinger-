import os
import time
import sys

from petsc4py import PETSc
rank = PETSc.COMM_WORLD.rank

import petsc4py
petsc4py.init(sys.argv)

from TDSE import Tdse
from Laser import laser
sys.path.append('/users/becker/dopl4670/Research/Atomic_Schrodinger/Common')
from Basis import basis
import utility

if rank == 0:
    if not os.path.exists("TDSE_files"):
        os.mkdir("TDSE_files")
    if not os.path.exists("temp"):
        os.mkdir("temp")
    if not os.path.exists("images"):
        os.mkdir("images")

if rank == 0:
    init_start = time.time()
    print("Initializing Simulation")
    print("\n")
input_file = "input.json"
TDSEInstance = Tdse(input_file)
TDSEInstance.readInState()
basisInstance = basis(TDSEInstance)
if rank == 0:
    init_end = time.time()
    print(f"Initializing Simulation Finished in {round(init_end - init_start, 2)} seconds")
    print("\n")

if rank == 0:
    laser_start = time.time()
    print("Constructing Laser Pulse")
    print("\n")
laserInstance = laser(TDSEInstance)
laserInstance.plotPulse()
if rank == 0:
    laser_end = time.time()
    print(f"Constructing Laser Pulse Finished in {round(laser_end - laser_start, 2)} seconds")
    print("\n")


if rank == 0:
    laser_start = time.time()
    print("Starting TDSE Simulation")
    print("\n")
TDSEInstance.readInState()
TDSEInstance.constructInteraction(basisInstance,laserInstance)
TDSEInstance.constructHHG(basisInstance,laserInstance)
TDSEInstance.constructAtomic()
TDSEInstance.propagateState(laserInstance)
if rank == 0:
    laser_end = time.time()
    print(f"TDSE Simulation Finished in {round(laser_end - laser_start, 2)} seconds")
    print("\n")

if rank == 0:
    print("Cleaning up...")
    os.system("rm -rf temp")
    os.system("mv TDSE.h5 TDSE_files")




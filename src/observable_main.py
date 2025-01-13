import sys as sys

import basis_class as basis
import pes_class as pes
import block_class as block

import laser_class as laser
import os as os
import time as time

input_file = "input.json"

from petsc4py import PETSc
from slepc4py import SLEPc
import petsc4py

rank = PETSc.COMM_WORLD.rank
size = PETSc.COMM_WORLD.size


if "PES" in sys.argv:
    os.system("mkdir PES_files")


    sim = pes.pes(input_file)
    bspline = basis.basis(sim)
    sim.loadS()
    sim.load_final_state()
    sim.project_out_bound_states()
    sim.expand_final_state(bspline)
    sim.compute_partial_spectra()
    sim.compute_angle_integrated()
    sim.compute_angle_resolved()

if "BLOCK" in sys.argv:
    LOG = "LOG" in sys.argv
    CONT = "CONT" in sys.argv
    print("LOG:",LOG)
    print("CONT:",CONT)

    dist = block.block("input.json")
    dist.loadS()
    dist.load_final_state()
    dist.compute_norm(CONT)
    dist.compute_distribution(projOutBound=CONT)






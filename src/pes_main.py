import sys as sys

import tise_class as tise
import tdse_class as tdse
import basis_class as basis
import pes_class as pes

import laser_class as laser
import os as os
import time as time

input_file = "input.json"

from petsc4py import PETSc
from slepc4py import SLEPc
import petsc4py

rank = PETSc.COMM_WORLD.rank
size = PETSc.COMM_WORLD.size



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



from collections import deque
import numpy as np
import resource

from petsc4py import PETSc
from slepc4py import SLEPc

comm = PETSc.COMM_WORLD
rank = comm.rank


def lm_expansion(lmax,state,polarization):
    delta_l = [1,-1]
    delta_m = []
    if polarization[0] or polarization[1]:
        delta_m.append(1)
        delta_m.append(-1)
    if polarization[2]:
        delta_m.append(0)

    def is_valid(l, m):
        return 0 <= l <= lmax and -l <= m <= l

    queue = deque([(0, 0)])
    reachable_points = set([(0, 0)])
    index_to_point = {}
    point_to_index = {}
    index = 0

    # Add initial state to dictionaries
    initial_l, initial_m = state[1], state[2]
    index_to_point[index] = (initial_l, initial_m)
    point_to_index[(initial_l, initial_m)] = index
    index += 1

    while queue:
        current_l, current_m = queue.popleft()
        for dl in delta_l:
            for dm in delta_m:
                new_l = current_l + dl
                new_m = current_m + dm
                if is_valid(new_l, new_m) and (new_l, new_m) not in reachable_points:
                    reachable_points.add((new_l, new_m))
                    queue.append((new_l, new_m))
                    
                    # Add to dictionaries
                    index_to_point[index] = (new_l, new_m)
                    point_to_index[(new_l, new_m)] = index
                    index += 1
    return point_to_index, index_to_point

def getNblock(dict):
    return len(dict)

def gather_matrix(M, comm, root):
        local_csr = M.getValuesCSR()
        local_indptr, local_indices, local_data = local_csr

        gathered_indices = comm.gather(local_indices, root=root)
        gathered_data = comm.gather(local_data, root=root)
        gathered_indptr = comm.gather(local_indptr, root=root)

        if comm.rank == root:
            global_indices = np.concatenate(gathered_indices).astype(np.int32)
            global_data = np.concatenate(gathered_data)
            global_indptr = [gathered_indptr[0]]
            offset = global_indptr[0][-1]
            for indptr in gathered_indptr[1:]:
                corrected_indptr = indptr[1:] + offset 
                global_indptr.append(corrected_indptr)
                offset += indptr[-1] - indptr[0]
            global_indptr = np.concatenate(global_indptr)
            return PETSc.Mat().createAIJWithArrays([M.getSize()[0], M.getSize()[1]], (global_indptr, global_indices, global_data), comm=PETSc.COMM_SELF)
        return None



def kron(A, B, comm,nonzeros):
    comm = comm.tompi4py()
    root = 0
    # Gather the matrices to the root
    rootA = gather_matrix(A, comm, root)
    rootB = gather_matrix(B, comm, root)

    if comm.rank == root:
        # Compute the Kronecker product only on the root
        rootC = rootA.kron(rootB)
        rootA.destroy()
        rootB.destroy()
        viewer = PETSc.Viewer().createBinary("temp/temp.bin","w",comm = PETSc.COMM_SELF)
        rootC.view(viewer)
        viewer.destroy()
    else:
        rootC = None
    
    C = PETSc.Mat(comm = PETSc.COMM_WORLD,nnz = nonzeros)
    viewer = PETSc.Viewer().createBinary('temp/temp.bin', 'r',comm = PETSc.COMM_WORLD)
    C.load(viewer)
    viewer.destroy()
    return C
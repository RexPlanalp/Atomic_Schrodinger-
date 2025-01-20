import numpy as np
import json as json
import pprint as pprint
import os as os
import matplotlib.pyplot as plt

from collections import deque
from petsc4py import PETSc
from slepc4py import SLEPc

rank = PETSc.COMM_WORLD.rank
size = PETSc.COMM_WORLD.size

class simulation:
    def __init__(self, json_file):
        self.input_params = self._load_json(json_file)
        self.setup_simulation()
    
    def _load_json(self, json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
        return self._dict_to_object(data)

    def _dict_to_object(self, data):
        if isinstance(data, dict):
            return {key: self._dict_to_object(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._dict_to_object(item) for item in data]
        else:
            return data

    def __getattr__(self, name):
        # Allow access through `input_params.key` directly
        if name in self.input_params:
            return self.input_params[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def _is_valid(self, l, m):
        return 0 <= l <= self.input_params["lm"]["lmax"] and -l <= m <= l and self.input_params["lm"]["mmin"] <= m <= self.input_params["lm"]["mmax"]
    
    def setup_simulation(self):
        self._components_ellipticity()
        self._lm_expansion()
        self._plot_lm_space()
        self._potential_map()
        self._A_0()
        self._time()
        self._space()
        if rank == 0:
            with open("sim_params.txt", "w") as file:
                pprint.pprint(self.input_params, stream=file)
        return True
   
    def _gather_matrix(self, M, comm, root):
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

    def kron(self, A, B, comm, nonzeros):
        comm = comm.tompi4py()
        root = 0
        # Gather the matrices to the root
        rootA = self._gather_matrix(A, comm, root)
        rootB = self._gather_matrix(B, comm, root)

        if comm.rank == root:
            # Compute the Kronecker product only on the root
            rootC = rootA.kron(rootB)
            rootA.destroy()
            rootB.destroy()
            viewer = PETSc.Viewer().createBinary("temp/temp.bin", "w", comm=PETSc.COMM_SELF)
            rootC.view(viewer)
            viewer.destroy()
        else:
            rootC = None
        
        C = PETSc.Mat(comm=PETSc.COMM_WORLD, nnz=nonzeros)
        viewer = PETSc.Viewer().createBinary('temp/temp.bin', 'r', comm=PETSc.COMM_WORLD)
        C.load(viewer)
        viewer.destroy()
        return C

    def _potential_map(self):
        def H(x):
                return (-1/(x+1E-25))  
                
        def He(x):
                return (-1/np.sqrt(x**2 + 1E-25)) + -1.0*np.exp(-2.0329*x)/np.sqrt(x**2 + 1E-25)  - 0.3953*np.exp(-6.1805*x)
        
        def Ar(x):
            return -1.0 / (x + 1e-25) - 17.0 * np.exp(-0.8103 * x) / (x + 1e-25) \
            - (-15.9583) * np.exp(-1.2305 * x) \
            - (-27.7467) * np.exp(-4.3946 * x) \
            - 2.1768 * np.exp(-86.7179 * x)

        potential_map = {
            "H": H,
            "He": He,
            "Ar": Ar
        }

        self.input_params["box"]["potential_map"] = potential_map
        self.input_params["box"]["potential_func"] = potential_map[self.input_params["species"]]
        return True

    def _components_ellipticity(self):
        polarization = self.input_params["laser"]["polarization"]
        polarization_norm = np.linalg.norm(polarization)
        if polarization_norm != 0:
            polarization /= np.linalg.norm(polarization)
        self.input_params["laser"]["polarization"] = polarization

        poynting = self.input_params["laser"]["poynting"]
        poynting_norm = np.linalg.norm(poynting)
        if poynting_norm != 0:
            poynting /= poynting_norm
        self.input_params["laser"]["poynting"] = poynting

        ellipticity = np.cross(polarization, poynting) 
        ellipticity_norm = np.linalg.norm(ellipticity)
        if ellipticity_norm != 0:
            ellipticity /= ellipticity_norm
        self.input_params["laser"]["ellipticity"]= ellipticity

        components = np.array([(1 if polarization[i] != 0 or self.input_params["laser"]["ell"] * ellipticity[i] != 0 else 0) for i in range(3)])
        self.input_params["laser"]["components"] = components
        

        return True

    def _lm_expansion(self):

        delta_l = [1, -1]
        delta_m = []

        if self.input_params["laser"]["components"][0] or self.input_params["laser"]["components"][1]:
            delta_m.extend([1, -1])
        if self.input_params["laser"]["components"][2]:
            delta_m.append(0)

        initial_l, initial_m = self.input_params["state"][1], self.input_params["state"][2]

        queue = deque([(initial_l, initial_m)])
        reachable_points = set([(initial_l, initial_m)])
        index_to_point = {}
        point_to_index = {}
        index = 0

        index_to_point[index] = (initial_l, initial_m)
        point_to_index[(initial_l, initial_m)] = index
        index += 1

        while queue:
            current_l, current_m = queue.popleft()
            for dl in delta_l:
                for dm in delta_m:
                    new_l = current_l + dl
                    new_m = current_m + dm
                    if self._is_valid(new_l, new_m) and (new_l, new_m) not in reachable_points:
                        reachable_points.add((new_l, new_m))
                        queue.append((new_l, new_m))
                        index_to_point[index] = (new_l, new_m)
                        point_to_index[(new_l, new_m)] = index
                        index += 1

        self.input_params["lm"]["lm_to_block"] = point_to_index
        self.input_params["lm"]["block_to_lm"] = index_to_point
        self.input_params["lm"]["n_blocks"] = len(point_to_index)
        return True

    def _A_0(self):
        I = self.input_params["laser"]["I"]
        w = self.input_params["laser"]["w"]
        E_0 = np.sqrt(I/3.51E16)
        A_0 = E_0 / w
        self.input_params["laser"]["A_0"] = A_0

    def _time(self):
        time_size = self.input_params["box"]["N"] * 2 * np.pi / self.input_params["laser"]["w"]
        post_time_size = self.input_params["box"]["N_post"] * 2 * np.pi / self.input_params["laser"]["w"]

        Nt = int(np.rint(time_size/ self.input_params["box"]["time_spacing"])) 
        Nt_post = int(np.rint(post_time_size/ self.input_params["box"]["time_spacing"]))

        self.input_params["box"]["Nt_total"] = Nt+Nt_post
        self.input_params["box"]["Nt"] = Nt
        self.input_params["box"]["Nt_post"] = Nt_post
        
        self.input_params["box"]["time_size"] = time_size
        self.input_params["box"]["t_range"] = np.array([0,time_size,Nt])
        self.input_params["box"]["t_post_range"] = np.array([time_size+self.input_params["box"]["time_spacing"],time_size+post_time_size,Nt_post])
        return True
    
    def _space(self):
        grid_spacing = self.input_params["box"]["grid_spacing"]
        grid_size = self.input_params["box"]["grid_size"]

        Nr = int(np.rint(grid_size / grid_spacing))
        self.input_params["box"]["r_range"] = np.array([grid_spacing, grid_size, Nr])
        return True

    def _plot_lm_space(self):
        if rank == 0:
            fig,ax = plt.subplots()
            space_size = self.input_params["lm"]["lmax"] + 1
            space = np.zeros((space_size, 2 * self.input_params["lm"]["lmax"] + 1))
            for l, m in self.input_params["lm"]["lm_to_block"].keys():
                space[self.input_params["lm"]["lmax"] - l, m + self.input_params["lm"]["lmax"]] = 1

            ax.imshow(np.flipud(space), cmap='gray', interpolation='none', origin='lower')
            ax.set_xlabel('m')
            ax.set_ylabel('l')
            ax.set_xticks([i for i in range(0, 2 * self.input_params["lm"]["lmax"] + 1, 10)])  # Positions for ticks
            ax.set_xticklabels([str(i - self.input_params["lm"]["lmax"]) for i in range(0, 2 * self.input_params["lm"]["lmax"] + 1, 10)])  # Labels from -lmax to lmax
            ax.set_title('Reachable (white) and Unreachable (black) Points in l-m Space')
            plt.savefig("images/lm_space.png")
            plt.clf()
            plt.close()
        


if __name__ == '__main__':
    if not os.path.exists("images"):
        os.makedirs("images")
    sim = simulation('input.json')
    
   



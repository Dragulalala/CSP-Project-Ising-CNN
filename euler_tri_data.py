import numpy as np
import os
from tqdm import tqdm
import sys
import multiprocessing
from numba import njit
import time

@njit
def fast_swendsen_wang_step_tri(spins, L, p, h_bonds, v_bonds, d_bonds, cluster_labels):
    for i in range(L):
        for j in range(L):
            h_bonds[i, j] = False
            v_bonds[i, j] = False
            d_bonds[i, j] = False
            cluster_labels[i, j] = 0

    # 1. Create bonds (Right, Down, and Down-Right for triangular topology)
    for i in range(L):
        for j in range(L):
            # Right neighbor
            r_i = i
            r_j = (j + 1) % L
            if spins[i, j] == spins[r_i, r_j]:
                if np.random.rand() < p:
                    h_bonds[i, j] = True
                    
            # Down neighbor
            d_i = (i + 1) % L
            d_j = j
            if spins[i, j] == spins[d_i, d_j]:
                if np.random.rand() < p:
                    v_bonds[i, j] = True

            # Diagonal neighbor (Down-Right)
            diag_i = (i + 1) % L
            diag_j = (j + 1) % L
            if spins[i, j] == spins[diag_i, diag_j]:
                if np.random.rand() < p:
                    d_bonds[i, j] = True

    current_label = 1
    
    stack_i = np.empty(L * L, dtype=np.int32)
    stack_j = np.empty(L * L, dtype=np.int32)
    
    # 2. Cluster building and flipping
    for i in range(L):
        for j in range(L):
            if cluster_labels[i, j] == 0:
                # Flip spin with 0.5 chance
                new_spin = 1 if np.random.rand() < 0.5 else -1
                
                # Mark initial site
                cluster_labels[i, j] = current_label
                spins[i, j] = new_spin
                
                # Initialize stack
                stack_i[0] = i
                stack_j[0] = j
                stack_ptr = 1
                
                # Walk
                while stack_ptr > 0:
                    stack_ptr -= 1
                    curr_i = stack_i[stack_ptr]
                    curr_j = stack_j[stack_ptr]
                    
                    # Right
                    right_j = (curr_j + 1) % L
                    if h_bonds[curr_i, curr_j] and cluster_labels[curr_i, right_j] == 0:
                        cluster_labels[curr_i, right_j] = current_label
                        spins[curr_i, right_j] = new_spin
                        stack_i[stack_ptr] = curr_i
                        stack_j[stack_ptr] = right_j
                        stack_ptr += 1
                        
                    # Left 
                    left_j = (curr_j - 1) % L
                    if h_bonds[curr_i, left_j] and cluster_labels[curr_i, left_j] == 0:
                        cluster_labels[curr_i, left_j] = current_label
                        spins[curr_i, left_j] = new_spin
                        stack_i[stack_ptr] = curr_i
                        stack_j[stack_ptr] = left_j
                        stack_ptr += 1
                        
                    # Down
                    down_i = (curr_i + 1) % L
                    if v_bonds[curr_i, curr_j] and cluster_labels[down_i, curr_j] == 0:
                        cluster_labels[down_i, curr_j] = current_label
                        spins[down_i, curr_j] = new_spin
                        stack_i[stack_ptr] = down_i
                        stack_j[stack_ptr] = curr_j
                        stack_ptr += 1
                        
                    # Up 
                    up_i = (curr_i - 1) % L
                    if v_bonds[up_i, curr_j] and cluster_labels[up_i, curr_j] == 0:
                        cluster_labels[up_i, curr_j] = current_label
                        spins[up_i, curr_j] = new_spin
                        stack_i[stack_ptr] = up_i
                        stack_j[stack_ptr] = curr_j
                        stack_ptr += 1

                    # Down-Right
                    dr_i = (curr_i + 1) % L
                    dr_j = (curr_j + 1) % L
                    if d_bonds[curr_i, curr_j] and cluster_labels[dr_i, dr_j] == 0:
                        cluster_labels[dr_i, dr_j] = current_label
                        spins[dr_i, dr_j] = new_spin
                        stack_i[stack_ptr] = dr_i
                        stack_j[stack_ptr] = dr_j
                        stack_ptr += 1

                    # Up-Left
                    ul_i = (curr_i - 1) % L
                    ul_j = (curr_j - 1) % L
                    if d_bonds[ul_i, ul_j] and cluster_labels[ul_i, ul_j] == 0:
                        cluster_labels[ul_i, ul_j] = current_label
                        spins[ul_i, ul_j] = new_spin
                        stack_i[stack_ptr] = ul_i
                        stack_j[stack_ptr] = ul_j
                        stack_ptr += 1
                        
                current_label += 1

    return L * L 


class IsingMC_SwendsenWang_Triangular:
    def __init__(self, length, temperature=0.):
        self.spins = np.ones((length, length), dtype=np.int32)
        self.L = length
        self.T = temperature
        self.M = length * length
        self.sw_prob = 0.0
        
        self.cluster_labels = np.zeros((length, length), dtype=np.int32) 
        self.h_bonds = np.zeros((length, length), dtype=np.bool_)
        self.v_bonds = np.zeros((length, length), dtype=np.bool_)
        self.d_bonds = np.zeros((length, length), dtype=np.bool_) # Diagonal bonds
        
        self.update_probabilities()
    
    def update_probabilities(self):
        if self.T > 0.:
            self.sw_prob = 1.0 - np.exp(-2.0 / self.T)
        else:
            self.sw_prob = 0.0
            
    def set_temperature(self, temperature):
        self.T = temperature
        self.update_probabilities()
    
    def reset_spins(self):
        self.spins.fill(1)
        self.M = self.L * self.L

    def swendsen_wang_step(self):
        return fast_swendsen_wang_step_tri(
            self.spins, self.L, self.sw_prob, 
            self.h_bonds, self.v_bonds, self.d_bonds, self.cluster_labels
        )


def simulate_temperature_worker(args):
    t_idx, temp, L, J, N, Nthermalization, Nsample, Nsubsweep = args
    sim = IsingMC_SwendsenWang_Triangular(L, temp)
    sim.reset_spins()

    # Thermalization Loop
    for _ in range(Nthermalization):
        spins_flipped = 0
        while spins_flipped < N: 
            spins_flipped += sim.swendsen_wang_step()

    # Storage array for this temperature's configurations
    configs = np.zeros((Nsample, N), dtype=np.int8)

    # First sample right after thermalization
    configs[0] = sim.spins.flatten()

    # Sampling Loop
    for n in range(1, Nsample):
        
        # Subsweep Loop to guarantee uncorrelated configurations
        spins_flipped = 0
        while spins_flipped < Nsubsweep:
            spins_flipped += sim.swendsen_wang_step()

        configs[n] = sim.spins.flatten()
    
    return temp, configs


if __name__ == '__main__':
    start_time = time.perf_counter()
    print("Starting Triangular Lattice Swendsen-Wang Configurations Generation...\n")

    # Global Parameters
    J = 1 
    temps = np.linspace(2.0, 5.0, 40)
    L_values = [10, 20, 30, 40, 60]
    
    Nthermalization = 100000
    Nsample = 250 # 250 uncorrelated configurations
    
    output_dir = 'CSPProject/CSP-Project-Ising-CNN/tri_data'
    os.makedirs(output_dir, exist_ok=True)

    try:
        usable_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        usable_cores = multiprocessing.cpu_count()
    
    print(f"Detected {usable_cores} usable CPU cores. Launching workers...\n")

    for L in L_values:
        print(f"--- Starting simulation for Triangular Lattice L = {L} ---")
        
        N = L**2
        Nsubsweep = N * 500 # Amount of work between samples to assure decorrelation

        tasks = [(t, temps[t], L, J, N, Nthermalization, Nsample, Nsubsweep) for t in range(len(temps))]

        with multiprocessing.Pool(processes=usable_cores) as pool:
            results = list(tqdm(pool.imap(simulate_temperature_worker, tasks), total=len(tasks), desc=f"L={L} Progress"))

        # Compile data to save to CSV
        print(f"\nAggregating data for L={L}...")
        
        # Format: [Temp, spin_0, spin_1, ..., spin_N-1]
        all_data_matrix = np.zeros((len(temps) * Nsample, N + 1), dtype=np.float32)
        
        row_idx = 0
        for temp, configs in results:
            for n in range(Nsample):
                all_data_matrix[row_idx, 0] = temp
                all_data_matrix[row_idx, 1:] = configs[n]
                row_idx += 1

        # File saving
        filename = os.path.join(output_dir, f'L{L}_tri.csv')
        
        # Determine formatting: temperature as float, spins as integers
        fmt_str = '%.6f' + ',' + ','.join(['%d'] * N)
        
        # Create a header row matching the CSV structure
        header = "Temperature," + ",".join([f"spin_{i}" for i in range(N)])

        np.savetxt(
            filename, 
            all_data_matrix, 
            fmt=fmt_str, 
            delimiter=',',
            header=header,
            comments='' 
        )
        print(f"Saved L={L} configurations to {filename}\n")

    end_time = time.perf_counter()
    print(f"Total Execution time: {end_time - start_time:.6f} seconds")
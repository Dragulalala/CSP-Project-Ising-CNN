import numpy as np
import random
import os
import csv
import multiprocessing
from tqdm import tqdm

class IsingMC_SwendsenWang_SquareIce:
    def __init__(self, length, temperature=0., J=1.0):
        self.L = length
        self.N_spins = 2 * length * length # Spins are on the bonds (2 per vertex)
        self.spins = np.ones(self.N_spins, dtype=int)
        self.T = temperature
        self.J = J
        self.sw_prob = 0.0
        
        # Graph structures for the medial lattice (bonds of the original lattice)
        self.spin_vertices = [None] * self.N_spins
        self.vertex_spins = [[] for _ in range(self.L * self.L)]
        self.adj = [set() for _ in range(self.N_spins)]
        
        self._build_lattice()
        self.update_probabilities()
    
    def _build_lattice(self):
        # Map spins to vertices
        for i in range(self.L):
            for j in range(self.L):
                v_curr = i * self.L + j
                v_right = i * self.L + (j + 1) % self.L
                v_down = ((i + 1) % self.L) * self.L + j
                
                # Indexing: 2*(i*L+j) for horizontal, 2*(i*L+j)+1 for vertical
                h_idx = 2 * (i * self.L + j)
                v_idx = 2 * (i * self.L + j) + 1
                
                self.spin_vertices[h_idx] = (v_curr, v_right)
                self.spin_vertices[v_idx] = (v_curr, v_down)

        # Build vertex -> spins mapping
        for s_idx, (v1, v2) in enumerate(self.spin_vertices):
            self.vertex_spins[v1].append(s_idx)
            self.vertex_spins[v2].append(s_idx)
            
        # Build adjacency graph for standard SW steps
        for v_spins in self.vertex_spins:
            for s1 in v_spins:
                for s2 in v_spins:
                    if s1 != s2:
                        self.adj[s1].add(s2)
                        
        self.adj = [list(neighbors) for neighbors in self.adj]

    def update_probabilities(self):
        if self.T == 0.:
            self.sw_prob = 1.0
        elif self.T == float('inf'):
            self.sw_prob = 0.0
        else:
            # Interaction energy difference for Q_v^2 cross terms
            self.sw_prob = 1.0 - np.exp(-4.0 * self.J / self.T)
            
    def set_temperature(self, temperature):
        self.T = temperature
        self.update_probabilities()
    
    def reset_spins(self):
        if self.T == 0.0:
            # Initialize to a valid Q_v = 0 Ice state
            self.spins[0::2] = 1   
            self.spins[1::2] = -1  
        else:
            self.spins = np.random.choice([-1, 1], size=self.N_spins)

    def loop_update_T0(self):
        """Loop-cluster update to navigate the degenerate Ice manifold at T=0."""
        start_v = random.randint(0, self.L * self.L - 1)
        curr_v = start_v
        target_val = 1 if random.random() < 0.5 else -1
        
        visited_vertices = {curr_v: 0}
        path_spins = []
        path_vertices = [curr_v]
        
        while True:
            candidates = [s for s in self.vertex_spins[curr_v] if self.spins[s] == target_val]
            if not candidates: 
                print("No candidates found for loop update, this should not happen in a perfect ice manifold.")
                break 
            
            chosen_s = random.choice(candidates)
            path_spins.append(chosen_s)
            
            v1, v2 = self.spin_vertices[chosen_s]
            next_v = v2 if v1 == curr_v else v1
            path_vertices.append(next_v)
            
            if next_v in visited_vertices:
                # Loop found, flip alternating spins
                loop_start = visited_vertices[next_v]
                loop_spins = path_spins[loop_start:]
                for s in loop_spins:
                    self.spins[s] *= -1
                break
                
            visited_vertices[next_v] = len(path_vertices) - 1
            curr_v = next_v
            target_val *= -1 

    def swendsen_wang_step(self):
        if self.T == 0.0:
            for _ in range(self.L): 
                self.loop_update_T0()
            return

        # WSK Cluster update for T > 0
        bonds = {i: [] for i in range(self.N_spins)}
        
        for i in range(self.N_spins):
            for j in self.adj[i]:
                if i < j:
                    if self.spins[i] != self.spins[j]: 
                        if np.random.rand() < self.sw_prob:
                            bonds[i].append(j)
                            bonds[j].append(i)

        cluster_labels = np.zeros(self.N_spins, dtype=int)
        current_label = 1
        
        for i in range(self.N_spins):
            if cluster_labels[i] == 0:
                new_spin = 1 if np.random.rand() < 0.5 else -1
                stack = [i]
                
                cluster_labels[i] = current_label
                self.spins[i] = new_spin
                
                while stack:
                    curr = stack.pop()
                    for neighbor in bonds[curr]:
                        if cluster_labels[neighbor] == 0:
                            cluster_labels[neighbor] = current_label
                            self.spins[neighbor] = new_spin
                            stack.append(neighbor)
                            
                current_label += 1

def generate_configuration_batch(args):
    """Worker function to generate a subset of configurations independently."""
    L, T, num_configs, decorrelation_steps = args
    
    sim = IsingMC_SwendsenWang_SquareIce(length=L, temperature=T)
    sim.reset_spins()
    
    # Thermalization
    for _ in range(20000):
        sim.swendsen_wang_step()
    
    configs = []
    for i in tqdm(range(num_configs), desc=f"L={L}, T={T}", position=0, leave=False):
        for _ in range(decorrelation_steps):
            sim.swendsen_wang_step()
        configs.append([T] + sim.spins.tolist())
        
    return configs

if __name__ == "__main__":
    lattice_sizes = [16]
    temperatures = [0.0, float('inf')]
    total_configs_per_temp = 5000
    decorrelation_steps = 500
    
    # Determine how many parallel workers we can use
    num_cores = multiprocessing.cpu_count()
    print(f"Detected {num_cores} CPU cores. Splitting workload...")

    data_folder = "CSPProject/CSP-Project-Ising-CNN/ice_data"
    os.makedirs(data_folder, exist_ok=True)
    
    # Start the pool
    with multiprocessing.Pool(processes=num_cores) as pool:
        for L in lattice_sizes:
            filename = os.path.join(data_folder, f"L{L}_ice.csv")
            print(f"\nStarting simulation for L={L}...")
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = ["Temperature"] + [f"s{i}" for i in range(2 * L * L)]
                writer.writerow(headers)
                
                for T in temperatures:
                    print(f"Processing configurations for T={T}")
                    
                    # Distribute the 5000 configs across the available cores evenly
                    configs_per_core = total_configs_per_temp // num_cores
                    remainder = total_configs_per_temp % num_cores
                    
                    tasks = []
                    for i in range(num_cores):
                        # Give the remainder out 1 by 1 to the first few cores
                        n_configs = configs_per_core + (1 if i < remainder else 0)
                        if n_configs > 0:
                            tasks.append((L, T, n_configs, decorrelation_steps))
                    
                    # Run workers in parallel with a tqdm progress bar
                    results = list(tqdm(
                        pool.imap(generate_configuration_batch, tasks), 
                        total=len(tasks), 
                        desc=f"Batches completed",
                        position=0,
                        leave=True
                    ))
                    
                    # Flatten the returned batches and write to CSV
                    for batch in results:
                        writer.writerows(batch)
                        
            print(f"Finished saving {total_configs_per_temp * len(temperatures)} configurations for L={L} to {filename}")

    print("\nAll simulations completed successfully.")
import numpy as np
import os

def split_npz(input_file):
    data = np.load(input_file)
    print(data["spins"].shape) 

    keys = list(data.keys())
    
    total_samples = len(data[keys[0]])
    half = total_samples // 2
    
    part1_data = {}
    part2_data = {}
    
    for key in keys:
        part1_data[key] = data[key][:half]
        part2_data[key] = data[key][half:]
        
    base_name = input_file.replace('.npz', '')
    np.savez_compressed(f"{base_name}_part1.npz", **part1_data)
    np.savez_compressed(f"{base_name}_part2.npz", **part2_data)
    
    print(f"Split {total_samples} samples into {half} and {total_samples - half}.")

split_npz("CSPProject/CSP-Project-Ising-CNN/tri_data_2/L60_tri.npz")
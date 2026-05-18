import numpy as np

def merge_npz(file1, file2, output_file):
    data1 = np.load(file1)
    data2 = np.load(file2)
    
    merged_data = {}
    

    for key in data1.keys():
        merged_data[key] = np.concatenate([data1[key], data2[key]], axis=0)
        
    np.savez_compressed(output_file, **merged_data)
    print(f"Successfully merged {file1} and {file2} into {output_file}")

merge_npz("CSPProject/CSP-Project-Ising-CNN/tri_data_2/L60_tri_part1.npz","CSPProject/CSP-Project-Ising-CNN/tri_data_2/L60_tri_part2.npz", "CSPProject/CSP-Project-Ising-CNN/tri_data_2/L60_tri2.npz")
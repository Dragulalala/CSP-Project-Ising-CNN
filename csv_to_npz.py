import numpy as np
import pandas as pd
import os
from tqdm import tqdm

input_dir = "C:/CSP/CSPProject/CSP-Project-Ising-CNN/tri_data_2/"
output_dir = "CSPProject/CSP-Project-Ising-CNN/tri_data_2/"
os.makedirs(output_dir, exist_ok=True)

for L in [ 10,20,30,40]:
    print(f"Processing L={L}...")

    csv_filename = os.path.join(input_dir, f"L{L}_tri.csv")

    total_rows = sum(1 for _ in open(csv_filename, "r", encoding="utf-8")) - 1

    temps_list = []
    spins_list = []

    for chunk in tqdm(pd.read_csv(csv_filename, chunksize=10000), total=(total_rows // 10000) + 1):
        temperatures_arr = chunk["Temperature"].to_numpy(dtype=np.float32)
        spins_arr = chunk.drop(columns=["Temperature"]).to_numpy(dtype=np.int8)

        temps_list.append(temperatures_arr)
        spins_list.append(spins_arr)

    temperatures_arr = np.concatenate(temps_list, axis=0)
    spins_arr = np.concatenate(spins_list, axis=0)

    filename = os.path.join(output_dir, f"L{L}_tri.npz")
    np.savez_compressed(filename, temperatures=temperatures_arr, spins=spins_arr)

    print(f"Saved {filename}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.optimize import brentq  # replaces fsolve

def load_npz_data(base_path, L):
    first_half_path  = os.path.join(base_path, f'L{L}_tri_first_half.npz')
    second_half_path = os.path.join(base_path, f'L{L}_tri_second_half.npz')
    single_path      = os.path.join(base_path, f'L{L}_tri.npz')

    if os.path.exists(first_half_path) and os.path.exists(second_half_path):
        print(f"  L={L}: loading split files...")
        d1 = np.load(first_half_path)
        d2 = np.load(second_half_path)
        temperatures = np.concatenate([d1["temperatures"], d2["temperatures"]], axis=0)
        spins        = np.concatenate([d1["spins"],        d2["spins"]],        axis=0)
    elif os.path.exists(single_path):
        print(f"  L={L}: loading single file...")
        d = np.load(single_path)
        temperatures = d["temperatures"]
        spins        = d["spins"]
    else:
        raise FileNotFoundError(f"No npz file(s) found for L={L} in {base_path}")
    return temperatures, spins


def process_binder_data(temperatures, spins):
    df = pd.DataFrame(spins)
    df['Temperature'] = temperatures
    spin_cols = [col for col in df.columns if col != 'Temperature']

    df['M']  = df[spin_cols].mean(axis=1)
    df['M2'] = df['M'] ** 2
    df['M4'] = df['M'] ** 4

    stats = df.groupby('Temperature').agg(
        mean_M2=('M2', 'mean'),
        mean_M4=('M4', 'mean')
    ).reset_index()

    stats['U_L'] = 1 - (stats['mean_M4'] / (3 * stats['mean_M2'] ** 2))
    return stats


def find_intersection_brentq(res1, res2, t_theory=3.640957):
    """
    Finds the Binder cumulant intersection robustly by scanning for a
    sign change in the difference curve, then bracketing with brentq.
    """
    t_min = max(res1['Temperature'].min(), res2['Temperature'].min())
    t_max = min(res1['Temperature'].max(), res2['Temperature'].max())

    f1 = interp1d(res1['Temperature'], res1['U_L'], kind='cubic', fill_value="extrapolate")
    f2 = interp1d(res2['Temperature'], res2['U_L'], kind='cubic', fill_value="extrapolate")

    def diff(t):
        return f1(t) - f2(t)

    # Scan for sign changes across the full overlapping range
    t_scan = np.linspace(t_min, t_max, 2000)
    d_scan = diff(t_scan)

    sign_changes = np.where(np.diff(np.sign(d_scan)))[0]

    if len(sign_changes) == 0:
        raise ValueError(
            f"No intersection found in T=[{t_min:.3f}, {t_max:.3f}]. "
            "Check that your curves actually cross in this range."
        )

    # If multiple crossings, pick the one closest to the theoretical Tc
    brackets = [(t_scan[i], t_scan[i + 1]) for i in sign_changes]
    best_bracket = min(brackets, key=lambda b: abs(np.mean(b) - t_theory))

    t_c = brentq(diff, best_bracket[0], best_bracket[1], xtol=1e-8)
    return t_c


# --- Configuration ---
TC_TRI    = 3.640957
L_sizes   = [10, 20, 30, 40, 60, 70, 80, 90, 100]
base_path = 'CSPProject/CSP-Project-Ising-CNN/tri_data_expanded/'

results_dict = {}
plt.figure(figsize=(10, 6))

for L in L_sizes:
    try:
        temperatures, spins = load_npz_data(base_path, L)
        res = process_binder_data(temperatures, spins)
        results_dict[L] = res
        plt.plot(res['Temperature'], res['U_L'], 'o-', label=f'L={L}', markersize=4)
    except FileNotFoundError as e:
        print(f"  Skipping L={L}: {e}")

# --- Calculating Tc via Intersection ---
if len(results_dict) >= 2:
    sizes = sorted(results_dict.keys())
    L1, L2 = sizes[-2], sizes[-1]

    try:
        t_c_calculated = find_intersection_brentq(results_dict[L1], results_dict[L2], t_theory=TC_TRI)

        plt.axvline(x=t_c_calculated, color='green', linestyle=':',
                    label=f'Calculated $T_c \\approx {t_c_calculated:.4f}$')

        print("-" * 30)
        print(f"Calculated Tc (Intersection of L={L1} and L={L2}): {t_c_calculated:.6f}")
        print(f"Theoretical Tc: {TC_TRI}")
        print(f"Error: {abs(t_c_calculated - TC_TRI):.6f}")
        print("-" * 30)

    except ValueError as e:
        print(f"Intersection error: {e}")

plt.axvline(x=TC_TRI, color='red', linestyle='--', alpha=0.5, label='Theory $T_c$')
plt.title('Binder Cumulant Intersection (Triangular Ising Model)')
plt.xlabel('Temperature ($T$)')
plt.ylabel('Binder Cumulant ($U_L$)')
plt.legend()
plt.grid(True)
plt.show()
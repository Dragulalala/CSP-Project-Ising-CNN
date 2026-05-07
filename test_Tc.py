import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.optimize import brentq  # replaces fsolve

def process_npz_binder(file_path):
    data = np.load(file_path)
    T_arr = data["temperatures"]
    s_arr = data["spins"]

    if s_arr.ndim == 3:
        M = np.mean(s_arr, axis=(1, 2))
    else:
        M = np.mean(s_arr, axis=1)

    M2 = M**2
    M4 = M**4

    temp_df = pd.DataFrame({'T': T_arr, 'M2': M2, 'M4': M4})
    stats = temp_df.groupby('T').agg(
        mean_M2=('M2', 'mean'),
        mean_M4=('M4', 'mean')
    ).reset_index()

    stats['U_L'] = 1 - (stats['mean_M4'] / (3 * stats['mean_M2']**2))
    return stats


def find_intersection_brentq(res1, res2):
    """
    Finds the intersection of two Binder cumulant curves robustly.
    Scans the overlapping temperature range for a sign change in the
    difference, then uses brentq to pin down the exact crossing.
    """
    # Overlapping temperature range only
    t_min = max(res1['T'].min(), res2['T'].min())
    t_max = min(res1['T'].max(), res2['T'].max())

    f1 = interp1d(res1['T'], res1['U_L'], kind='cubic', fill_value="extrapolate")
    f2 = interp1d(res2['T'], res2['U_L'], kind='cubic', fill_value="extrapolate")

    def diff(t):
        return f1(t) - f2(t)

    # Scan for sign changes across the overlapping range
    t_scan = np.linspace(t_min, t_max, 2000)
    d_scan = diff(t_scan)

    sign_changes = np.where(np.diff(np.sign(d_scan)))[0]

    if len(sign_changes) == 0:
        raise ValueError(
            f"No intersection found in T=[{t_min:.3f}, {t_max:.3f}]. "
            "Check that your curves actually cross in this range."
        )

    # Use the sign change closest to the theoretical Tc as the bracket
    brackets = [(t_scan[i], t_scan[i + 1]) for i in sign_changes]
    best_bracket = min(brackets, key=lambda b: abs(np.mean(b) - TC_SQUARE))

    t_c = brentq(diff, best_bracket[0], best_bracket[1], xtol=1e-8)
    return t_c


# --- Configuration ---
L_sizes   = [10, 20, 30, 40, 60, 70, 80, 90, 100]
base_path = 'CSPProject/CSP-Project-Ising-CNN/data_decorr/'

TC_SQUARE    = 2 / np.log(1 + np.sqrt(2))
results_dict = {}

plt.figure(figsize=(10, 6))

for L in L_sizes:
    full_path = os.path.join(base_path, f'L{L}_ising.npz')
    if os.path.exists(full_path):
        print(f"Loading L={L}...")
        res = process_npz_binder(full_path)
        results_dict[L] = res
        plt.plot(res['T'], res['U_L'], 'o-', label=f'L={L}', markersize=4)

# --- Tc via Intersection ---
if len(results_dict) >= 2:
    sizes = sorted(results_dict.keys())
    L1, L2 = sizes[-2], sizes[-1]

    try:
        t_c_calculated = find_intersection_brentq(results_dict[L1], results_dict[L2])

        print("\n" + "=" * 40)
        print(f"SQUARE ISING MODEL ANALYSIS")
        print(f"Calculated Tc (L={L1} & L={L2}): {t_c_calculated:.6f}")
        print(f"Theoretical Tc:                  {TC_SQUARE:.6f}")
        print(f"Error:                           {abs(t_c_calculated - TC_SQUARE):.6f}")
        print("=" * 40)

        plt.axvline(x=t_c_calculated, color='green', linestyle=':',
                    label=f'Calculated $T_c \\approx {t_c_calculated:.3f}$')

    except ValueError as e:
        print(f"Intersection error: {e}")

plt.axvline(x=TC_SQUARE, color='red', linestyle='--', alpha=0.5, label='Theory $T_c$')
plt.title('Binder Cumulant Intersection (Square Ising Model)')
plt.xlabel('Temperature ($T$)')
plt.ylabel('Binder Cumulant ($U_L$)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()
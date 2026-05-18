import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# ── Configuration ──────────────────────────────────────────────────────────────
TC_TRI    = 3.640957          # exact Tc/J for triangular Ising = 4/ln(3)
TC        = 2.269185          # square lattice Tc/J = 2/ln(1+√2) ≈ 2.269185
L_sizes   = [10, 20, 30, 40, 60]
base_path = 'CSPProject/CSP-Project-Ising-CNN/tri_data_expanded/'
N_BOOTSTRAP = 200             # bootstrap resamples for error estimation

# ── Data loading (unchanged from your code) ────────────────────────────────────
def load_npz_data(base_path, L):
    first_half_path  = os.path.join(base_path, f'L{L}_tri_first_half.npz')
    second_half_path = os.path.join(base_path, f'L{L}_tri_second_half.npz')
    single_path      = os.path.join(base_path, f'L{L}_tri.npz')

    if os.path.exists(first_half_path) and os.path.exists(second_half_path):
        print(f"  L={L}: loading split files...")
        d1 = np.load(first_half_path)
        d2 = np.load(second_half_path)
        return (np.concatenate([d1["temperatures"], d2["temperatures"]]),
                np.concatenate([d1["spins"],        d2["spins"]]))
    elif os.path.exists(single_path):
        print(f"  L={L}: loading single file...")
        d = np.load(single_path)
        return d["temperatures"], d["spins"]
    else:
        raise FileNotFoundError(f"No npz file(s) found for L={L} in {base_path}")


# ── Binder cumulant from raw spin array ───────────────────────────────────────
def compute_binder(temperatures, spins):
    unique_T = np.sort(np.unique(temperatures))
    results = []

    for t in unique_T:
        mask = temperatures == t
        S = spins[mask]
        M = S.mean(axis=1)
        M2 = M**2
        M4 = M**4

        # Mean values
        m2_avg = np.mean(M2)
        m4_avg = np.mean(M4)
        U = 1.0 - m4_avg / (3.0 * m2_avg**2)

        # Simple SEM (Standard Error of the Mean)
        # Note: This ignores correlation between M2 and M4 for simplicity
        err_m2 = np.std(M2, ddof=1) / np.sqrt(len(M2))
        err_m4 = np.std(M4, ddof=1) / np.sqrt(len(M4))

        # Partial derivatives for Delta Method
        # dU/dm4 = -1 / (3 * m2^2)
        # dU/dm2 = 2 * m4 / (3 * m2^3)
        deriv_m4 = -1.0 / (3.0 * m2_avg**2)
        deriv_m2 = (2.0 * m4_avg) / (3.0 * m2_avg**3)

        U_err = np.sqrt((deriv_m4 * err_m4)**2 + (deriv_m2 * err_m2)**2)
        
        results.append({'Temperature': t, 'U_L': U, 'U_L_err': U_err})

    return pd.DataFrame(results)

# ── Crossing temperature between two Binder curves ────────────────────────────
def find_crossing(df1, df2, t_theory=TC_TRI):
    t_min = max(df1['Temperature'].min(), df2['Temperature'].min())
    t_max = min(df1['Temperature'].max(), df2['Temperature'].max())

    f1 = interp1d(df1['Temperature'], df1['U_L'], kind='cubic')
    f2 = interp1d(df2['Temperature'], df2['U_L'], kind='cubic')

    # Find the crossing point T*
    def diff(t): return f1(t) - f2(t)
    
    t_scan = np.linspace(t_min, t_max, 1000)
    idx = np.where(np.diff(np.sign(diff(t_scan))))[0]
    if len(idx) == 0: raise ValueError("No crossing found")
    
    T_cross = brentq(diff, t_scan[idx[0]], t_scan[idx[0]+1])

    # 1. Get slopes (numerical derivative) at T_cross
    dt = 1e-5
    s1 = (f1(T_cross + dt) - f1(T_cross - dt)) / (2 * dt)
    s2 = (f2(T_cross + dt) - f2(T_cross - dt)) / (2 * dt)

    # 2. Get vertical errors at T_cross (interpolated)
    err_func1 = interp1d(df1['Temperature'], df1['U_L_err'], kind='linear')
    err_func2 = interp1d(df2['Temperature'], df2['U_L_err'], kind='linear')
    
    sig1 = err_func1(T_cross)
    sig2 = err_func2(T_cross)

    # 3. Horizontal error formula
    T_err = np.sqrt(sig1**2 + sig2**2) / abs(s1 - s2)

    return T_cross, T_err

# ══════════════════════════════════════════════════════════════════════════════
# MAIN: load data, compute Binder cumulants, find pairwise crossings
# ══════════════════════════════════════════════════════════════════════════════
binder_dict  = {}
available_Ls = []

for L in L_sizes:
    try:
        T_arr, spins = load_npz_data(base_path, L)
        print(f"  Computing Binder cumulant for L={L} …")
        binder_dict[L] = compute_binder(T_arr, spins)
        available_Ls.append(L)
    except FileNotFoundError as e:
        print(f"  Skipping L={L}: {e}")

available_Ls = sorted(available_Ls)
print(f"\nAvailable system sizes: {available_Ls}")

# ── Pairwise crossing temperatures ────────────────────────────────────────────
crossing_Ts   = []
crossing_errs = []
crossing_Ls   = []   # representative L for the x-axis

for i in range(len(available_Ls) - 1):
    L1, L2 = available_Ls[i], available_Ls[i+1]
    print(f"\n  Finding crossing: L={L1} × L={L2} …")
    try:
        T_c, T_err = find_crossing(binder_dict[L1], binder_dict[L2])
        L_rep      = 2.0 / (1.0/L1 + 1.0/L2)   # harmonic mean
        crossing_Ts.append(T_c)
        crossing_errs.append(T_err)
        crossing_Ls.append(L_rep)
        print(f"    T* = {T_c:.5f} ± {T_err:.5f}  (L_rep = {L_rep:.1f})")
    except ValueError as e:
        print(f"    Skipping: {e}")

crossing_Ts   = np.array(crossing_Ts)
crossing_errs = np.array(crossing_errs)
inv_L         = 1.0 / np.array(crossing_Ls)

# ── Linear extrapolation 1/L → 0 (Updated with exact error calculation) ───────
valid   = ~np.isnan(crossing_errs)
weights = 1.0 / crossing_errs[valid]**2          # inverse-variance weighting
coeffs, cov = np.polyfit(inv_L[valid], crossing_Ts[valid], 1, w=weights, cov=True)
x_fit   = np.linspace(0, inv_L.max() * 1.08, 300)
y_fit   = np.polyval(coeffs, x_fit)

Tc_extrap = coeffs[1]
Tc_err = np.sqrt(cov[1, 1])  # Standard Deviation of intercept fit parameter

print(f"\n{'='*50}")
print(f"Extrapolated Tc (1/L → 0) : {Tc_extrap:.5f} ± {Tc_err:.5f} (SD of fit)")
print(f"Exact Tc                   : {TC_TRI:.5f}")
print(f"Difference                 : {abs(Tc_extrap - TC_TRI):.5f}")
print(f"{'='*50}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: two panels — Binder cumulants (left) + panel-f style FSS (right)
# ══════════════════════════════════════════════════════════════════════════════
colors  = plt.cm.viridis(np.linspace(0.1, 0.9, len(available_Ls)))
markers = ['x', '^', 'o', 'D', 's', 'P', '*', 'v', 'h']

fig, (ax_b, ax_f) = plt.subplots(1, 2, figsize=(14, 5))

# ── Left: Binder cumulant curves ──────────────────────────────────────────────
for j, L in enumerate(available_Ls):
    df = binder_dict[L]
    ax_b.errorbar(df['Temperature'], df['U_L'], yerr=df['U_L_err'],
                  marker=markers[j % len(markers)], markersize=4,
                  linestyle='-', linewidth=1.2, color=colors[j],
                  capsize=2, elinewidth=0.8, label=f'L = {L}')

ax_b.axvline(x=TC_TRI,    color='darkorange', linewidth=1.8, linestyle='-',
             label=f'Exact $T_c/J = {TC_TRI:.4f}$')
ax_b.axvline(x=Tc_extrap, color='steelblue',  linewidth=1.5, linestyle='--',
             label=f'Extrap. $T_c/J = {Tc_extrap:.4f}$')
ax_b.set_xlabel('$T/J$', fontsize=13)
ax_b.set_ylabel('Binder Cumulant $U_L$', fontsize=13)
ax_b.legend(fontsize=8, framealpha=0.7)
ax_b.grid(True, alpha=0.25)

# ── Right: panel-f style FSS (Updated to visualize standard deviation) ────────
ax_f.errorbar(inv_L, crossing_Ts, yerr=crossing_errs,
              marker='v', linestyle='-', color='crimson',
              markersize=8, capsize=4, elinewidth=1.2, linewidth=1.5,
              label=r'$T^*/J$ (Binder crossing)')

ax_f.plot(x_fit, y_fit, 'b--', linewidth=1.5, alpha=0.8,
          label=fr'Weighted extrap. $T_c/J = {Tc_extrap:.4f} \pm {Tc_err:.4f}$')

ax_f.axhline(y=TC_TRI, color='darkorange', linewidth=1.8, linestyle='-',
             label=fr'Exact $T_c/J = {TC_TRI:.4f}$')

# Plot the extrapolated intercept with its standard deviation (SD) error bounds at 1/L = 0
ax_f.errorbar(0, Tc_extrap, yerr=Tc_err, marker='o', color='steelblue', 
              markersize=6, capsize=4, elinewidth=1.5, label='Extrapolated $T_c$')

# Annotate which L pairs each point represents
for k, (x, y, L1, L2) in enumerate(zip(
        inv_L, crossing_Ts,
        available_Ls[:-1], available_Ls[1:])):
    ax_f.annotate(f'({L1},{L2})', xy=(x, y), xytext=(4, 4),
                  textcoords='offset points', fontsize=7, color='dimgray')

ax_f.set_xlabel('$1/L_{\\mathrm{harm}}$', fontsize=13)
ax_f.set_ylabel('$T^*/J$', fontsize=13)
ax_f.set_xlim(-0.002, inv_L.max() * 1.15)
ax_f.legend(fontsize=9, framealpha=0.7)
ax_f.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('CSPProject/CSP-Project-Ising-CNN/binder_fss_triangular.pdf', dpi=900, bbox_inches='tight')
plt.show()
print("Saved to binder_fss_triangular.png")
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tensorflow.keras.models import load_model
from scipy.optimize import curve_fit

# ── Critical parameters ────────────────────────────────────────────────────────
T_c = 2.269185          # exact Tc/J for 2-D square-lattice Ising = 2/ln(1+√2)
T_c = 3.64095           # triangular lattice Tc/J = 4/ln(3) ≈ 3.64095
nu  = 1.0               # correlation-length exponent

L_sizes = [10, 20, 30, 40, 60]

markers    = ['x', '^', 'o', 'D', 's']
markersizes = [7,   6,   6,   6,   6]

# Blue gradient → ordered neuron 
# Red  gradient → disordered neuron
blue_cmap = plt.cm.Blues(np.linspace(0.40, 0.90, len(L_sizes)))
red_cmap  = plt.cm.Reds( np.linspace(0.40, 0.90, len(L_sizes)))

# ── Figure layout ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax_d, ax_e, ax_f = axes

# Storage for panel-f data
crossing_Ts   = []
crossing_errs = []

# ── Main loop over system sizes ────────────────────────────────────────────────
for i, L in enumerate(L_sizes):

    # ── Load model & data ──────────────────────────────────────────────────────
    model = load_model(
        f"CSPProject/CSP-Project-Ising-CNN/models_100/ising_classifier_L{L}.h5"
    )
    data    = np.load(
        f"CSPProject/CSP-Project-Ising-CNN/tri_data_2/L{L}_tri.npz"
    )
    T       = data["temperatures"]
    configs = data["spins"]

    print(f"Config shape for L={L}: {configs.shape}")


    # ── Predictions ────────────────────────────────────────────────────────────
    preds            = model.predict(configs, batch_size=512, verbose=0)
    disordered_out   = preds[:, 0]   # neuron 0: disordered
    ordered_out      = preds[:, 1]   # neuron 1: ordered

    # ── Average & SEM per temperature ──────────────────────────────────────────
    # ERROR CALCULATION: Standard Error of the Mean 1.96*(SEM) = std / sqrt(n)
    unique_T     = np.sort(np.unique(T))
    avg_ord      = np.empty(len(unique_T))
    avg_dis      = np.empty(len(unique_T))
    sem_ord      = np.empty(len(unique_T))
    sem_dis      = np.empty(len(unique_T))

    for j, t_val in enumerate(unique_T):
        mask          = (T == t_val)
        n             = mask.sum()
        avg_ord[j]    = np.mean(ordered_out[mask])
        avg_dis[j]    = np.mean(disordered_out[mask])
        sem_ord[j]    = 1.96 * np.std(ordered_out[mask],    ddof=1) / np.sqrt(n)
        sem_dis[j]    = 1.96 * np.std(disordered_out[mask], ddof=1) / np.sqrt(n)

    print(f"L={L:3d}: {len(unique_T)} temperature points, "
          f"T ∈ [{unique_T.min():.3f}, {unique_T.max():.3f}]")

    # ── Find crossing temperature T* (ordered = disordered ≈ 0.5) ─────────────
    diff = avg_ord - avg_dis                  # positive below Tc, negative above
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    # Check if neurons sum to 1
    prob_sum = avg_ord + avg_dis
    mean_sum = np.mean(prob_sum)
    std_sum = np.std(prob_sum)

    print(f"L={L:3d}: Mean Probability Sum = {mean_sum:.6f} (± {std_sum:.6f})")

    if not np.isclose(mean_sum, 1.0, atol=1e-3):
        print("Warning: Neurons do not sum to 1. Using intersection point instead of 0.5.")

    if len(sign_changes) > 0:
        idx = sign_changes[0]
        # Linear interpolation between the two bracketing points
        dT      = unique_T[idx+1] - unique_T[idx]
        T_star  = unique_T[idx] - diff[idx] * dT / (diff[idx+1] - diff[idx])
        # error in Critical temperature
        slopes = np.gradient(diff, unique_T)
        slope_at_Tstar = np.interp(T_star, unique_T, slopes)
        combined_sem = np.sqrt(sem_ord**2 + sem_dis**2)
        sem_at_Tstar = np.interp(T_star, unique_T, combined_sem)
        T_err = np.abs(sem_at_Tstar / slope_at_Tstar)
        #T_err   = dT / 2.0

    else:
        T_star, T_err = np.nan, np.nan

    crossing_Ts.append(T_star)
    crossing_errs.append(T_err)

    # ── Scaling variable: t = (T − Tc)/J  (J ≡ 1), x = t·L^(1/ν) ────────────
    t_reduced = unique_T - T_c
    scaled_x  = t_reduced * (L ** (1.0 / nu))

    kw_ord = dict(marker=markers[i], markersize=markersizes[i],
                  color=blue_cmap[i], capsize=2, elinewidth=0.8,
                  linewidth=1.2)
    kw_dis = dict(marker=markers[i], markersize=markersizes[i],
                  color=red_cmap[i],  capsize=2, elinewidth=0.8,
                  linewidth=1.2)

    # ── Panel d: output vs T/J ────────────────────────────────────────────────
    lbl = f'L = {L}'
    ax_d.errorbar(unique_T, avg_ord, yerr=sem_ord,
                  linestyle='-', label=lbl, **kw_ord)
    ax_d.errorbar(unique_T, avg_dis, yerr=sem_dis,
                  linestyle='-', **kw_dis)

    # ── Panel e: data collapse ────────────────────────────────────────────────
    ax_e.errorbar(scaled_x, avg_ord, yerr=sem_ord,
                  linestyle='None', **kw_ord)
    ax_e.errorbar(scaled_x, avg_dis, yerr=sem_dis,
                  linestyle='None', **kw_dis)
    
    


# ── Panel d styling ────────────────────────────────────────────────────────────
ax_d.axvline(x=T_c, color='darkorange', linestyle='-', linewidth=1.8,
             zorder=5, label=f'$T_c/J = {T_c:.4f}$')
ax_d.set_xlabel('$T/J$', fontsize=13)
ax_d.set_ylabel('Output layer', fontsize=13)
ax_d.set_ylim(-0.03, 1.03)
ax_d.set_xlim(unique_T.min() - 0.05, unique_T.max() + 0.05)
ax_d.legend(fontsize=9, framealpha=0.7)
ax_d.text(0.03, 0.97, 'a', transform=ax_d.transAxes,
          fontsize=14, fontweight='bold', va='top')
ax_d.grid(True, alpha=0.25)

# ── Panel e styling ────────────────────────────────────────────────────────────
ax_e.axvline(x=0, color='k', linestyle='--', alpha=0.35, linewidth=1)
ax_e.set_xlabel(r'$tL^{1/\nu}$', fontsize=13)
ax_e.set_ylabel('Output layer', fontsize=13)
ax_e.set_ylim(-0.03, 1.03)
ax_e.set_xlim(-10, 10)
ax_e.text(0.03, 0.97, 'b', transform=ax_e.transAxes,
          fontsize=14, fontweight='bold', va='top')
ax_e.grid(True, alpha=0.25)

# ── Panel f: finite-size scaling of T* ────────────────────────────────────────
crossing_Ts   = np.array(crossing_Ts)
crossing_errs = np.array(crossing_errs)
inv_L         = 1.0 / np.array(L_sizes)

ax_f.errorbar(inv_L, crossing_Ts, yerr=crossing_errs,
              marker='v', linestyle='-', color='crimson',
              markersize=7, capsize=3, elinewidth=1.0, linewidth=1.5,
              label=r'$T^*/J$')

# Linear extrapolation 1/L → 0
# 1. Define the linear model for FSS: Tc(L) = Tc_inf + a * (1/L)
def fss_model(inv_L, tc_inf, slope):
    return tc_inf + slope * inv_L

# 2. Filter valid data (removing NaNs)
valid = ~np.isnan(crossing_Ts)
x_data = inv_L[valid]
y_data = np.array(crossing_Ts)[valid]
y_errs = np.array(crossing_errs)[valid] 

if len(x_data) >= 2:
    # 3. Perform the Weighted Least Squares fit
    # sigma=y_errs tells the fit to trust points with smaller errors more
    popt, pcov = curve_fit(fss_model, x_data, y_data, sigma=y_errs, absolute_sigma=False)
    
    tc_extrap = popt[0]
    slope_fit = popt[1]
    
    # 4. Extract the error on the intercept (Tc_inf)
    tc_extrap_err = np.sqrt(pcov[0, 0])
    
    # 5. Plotting
    x_fit = np.linspace(0, inv_L.max() * 1.05, 200)
    y_fit = fss_model(x_fit, *popt)
    
    # Updated label with the error
    label_text = fr'Extrap. $T_c/J = {tc_extrap:.4f} \pm {tc_extrap_err:.4f}$'
    
    ax_f.plot(x_fit, y_fit, 'b--', linewidth=1.4, alpha=0.7, label=label_text)
        

# Exact Tc (orange horizontal line, matching panel d)
ax_f.axhline(y=T_c, color='darkorange', linestyle='-', linewidth=1.8,
             label=fr'Exact $T_c/J={T_c:.4f}$')

ax_f.set_xlabel('$1/L$', fontsize=13)
ax_f.set_ylabel('$T^*/J$', fontsize=13)
ax_f.set_xlim(-0.002, inv_L.max() * 1.08)
ax_f.legend(fontsize=9, framealpha=0.7)
ax_f.text(0.03, 0.97, 'c', transform=ax_f.transAxes,
          fontsize=14, fontweight='bold', va='top')
ax_f.grid(True, alpha=0.25)

plt.suptitle(
    r'Finite-size scaling of CNN outputs — 2-D Ising model (square lattice)',
    fontsize=13, y=1.01
)
plt.tight_layout()
plt.savefig('CSPProject/CSP-Project-Ising-CNN/data_collapse_tri_2_1.96sem.pdf', dpi=900, bbox_inches='tight')
plt.show()
print("Done. Figure saved to data_collapse")
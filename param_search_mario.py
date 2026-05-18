import numpy as np
import tensorflow as tf
print(tf.__version__)
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
from pathlib import Path

DATA_DIR      = "data_decorr"
DATA_TEST_DIR = "data_test"
OUT_DIR       = Path("param_search")

L_SIZES = [10, 20, 30, 40, 60]

# ── L2-only search ─────────────────────────────────────────────────────────────
L2_VALUES = [1e-4, 7e-4, 1e-3, 5e-3, 1e-2, 3e-2, 5e-2, 1e-1, 3e-1, 1.0]

# ── Previous grids (kept for reference) ───────────────────────────────────────
# LAM_VAR_VALUES = [0.0, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 1.0]  # v3 (no-l2)
# L2_VALUES_2D   = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]                           # v2 (2D)
# ──────────────────────────────────────────────────────────────────────────────

SWEEP_EPOCHS   = 80
SWEEP_PATIENCE = 10
BEST_EPOCHS    = 150
BEST_PATIENCE  = 15

NCOLS = 5  # columns in the per-L hidden-args grid


def build_model(config_size, l2):
    x = tf.keras.Input((config_size,))
    y = tf.keras.layers.Dense(
        100,
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
        kernel_regularizer=tf.keras.regularizers.l2(l2),
    )(x)
    z = tf.keras.layers.Dense(2, activation='softmax')(y)
    model = tf.keras.Model(inputs=x, outputs=z)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def compute_mean_R2(model, val_conf):
    """Mean R² of a linear fit h_j ~ m across all hidden neurons."""
    weights, bias = model.layers[1].get_weights()
    m = np.mean(val_conf, axis=1)
    h = val_conf @ weights + bias
    A = np.column_stack([m, np.ones(len(m))])
    R2_list = []
    for j in range(h.shape[1]):
        y = h[:, j]
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        ss_res = np.sum((y - A @ coeffs) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        R2_list.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    return float(np.mean(R2_list))


def train(train_conf, train_label, val_conf, val_label, config_size, l2, epochs, patience):
    model = build_model(config_size, l2)
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                          patience=max(5, patience // 3), min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=patience,
                      restore_best_weights=True),
    ]
    history = model.fit(
        train_conf, train_label,
        validation_data=(val_conf, val_label),
        batch_size=32, epochs=epochs, callbacks=callbacks, verbose=0,
    )
    val_loss = float(min(history.history['val_loss']))
    mean_R2  = compute_mean_R2(model, val_conf)
    return model, val_loss, mean_R2


def load_split(L):
    data    = np.load(f"{DATA_DIR}/L{L}_ising.npz")
    T       = data["temperatures"]
    T_c     = 2 / np.log(1 + np.sqrt(2))
    labels  = np.transpose(np.array([(T > T_c).astype(int), (T < T_c).astype(int)]))
    configs = data["spins"]
    idx     = np.random.default_rng(seed=42).permutation(len(T))
    T, configs, labels = T[idx], configs[idx], labels[idx]
    train_conf, val_conf   = np.split(configs, [80000])
    train_label, val_label = np.split(labels,  [80000])
    return train_conf, train_label, val_conf, val_label


def fmt(v):
    return '0' if v == 0.0 else f'{v:.0e}'


def add_row_heatmap(ax, row, ticklabels, title, cmap, cbar_label):
    grid = row[np.newaxis, :]
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    im = ax.imshow(grid, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(ticklabels))); ax.set_xticklabels(ticklabels, fontsize=8)
    ax.set_yticks([]); ax.set_xlabel('l2', fontsize=8)
    ax.set_title(title, fontsize=9)
    mid = (vmin + vmax) / 2
    for j, v in enumerate(row):
        if not np.isnan(v):
            ax.text(j, 0, f'{v:.4f}', ha='center', va='center',
                    color='black' if v < mid else 'white', fontsize=7)
    plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)


OUT_DIR.mkdir(parents=True, exist_ok=True)
results_file = OUT_DIR / "results_only_l2.json"

try:
    all_results = json.loads(results_file.read_text())
except (FileNotFoundError, json.JSONDecodeError):
    all_results = {}

best_params = {}

# ── Grid search ────────────────────────────────────────────────────────────────
for L in L_SIZES:
    print(f"\n{'='*46}\nGrid search  L={L}\n{'='*46}")
    train_conf, train_label, val_conf, val_label = load_split(L)
    config_size = train_conf.shape[1]

    data_test = np.load(f"{DATA_TEST_DIR}/L{L}_ising.npz")
    m_test    = np.mean(data_test['spins'], axis=1)

    key = str(L)
    all_results.setdefault(key, {})

    val_losses   = np.full(len(L2_VALUES), np.nan)
    mean_R2s     = np.full(len(L2_VALUES), np.nan)
    stored_wb    = []  # (weights, bias) per l2 for hidden-args grid

    for i, l2 in enumerate(L2_VALUES):
        combo_key = str(l2)
        entry = all_results[key].get(combo_key, {})

        if isinstance(entry, dict) and entry.get("mean_R2") is not None:
            
            # Need the model to get weights — retrain even if metrics are cached
            model, val_loss, mean_R2 = train(train_conf, train_label, val_conf, val_label,
                                config_size, l2, SWEEP_EPOCHS, SWEEP_PATIENCE)
            val_losses[i] = val_loss
            mean_R2s[i]   = mean_R2
            print(f"  [cached] l2={fmt(l2):6s}"
                  f"  loss={val_losses[i]:.5f}  R²={mean_R2s[i]:.4f}")
        else:
            model, val_loss, mean_R2 = train(
                train_conf, train_label, val_conf, val_label,
                config_size, l2, SWEEP_EPOCHS, SWEEP_PATIENCE)
            val_losses[i] = val_loss
            mean_R2s[i]   = mean_R2
            all_results[key][combo_key] = {"val_loss": val_loss, "mean_R2": mean_R2}
            print(f"  l2={fmt(l2):6s}"
                  f"  loss={val_loss:.5f}  R²={mean_R2:.4f}")
            results_file.write_text(json.dumps(all_results))

        stored_wb.append(model.layers[1].get_weights())

    composite = val_losses * (2.0 - mean_R2s)
    xl        = [fmt(v) for v in L2_VALUES]

    # ── 3-panel heatmap ────────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 3, figsize=(18, 2.5))
    add_row_heatmap(axs[0], val_losses, xl, f'Val loss — L={L}',  'viridis_r', 'val loss')
    add_row_heatmap(axs[1], mean_R2s,   xl, f'Mean R² — L={L}',   'viridis',   'mean R²')
    add_row_heatmap(axs[2], composite,  xl, f'Composite — L={L}', 'viridis_r', 'loss × (2 − R²)')
    plt.suptitle(f'L={L}  (L2 only)', fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'heatmap_L{L}_v3.png', dpi=120)
    plt.close()

    # ── Hidden args grid (all l2 values) ──────────────────────────────────────
    n      = len(L2_VALUES)
    nrows  = (n + NCOLS - 1) // NCOLS
    fig, axs_grid = plt.subplots(nrows, NCOLS, figsize=(NCOLS * 4, nrows * 3))
    axs_flat = np.array(axs_grid).flatten()

    for i, (l2, (weights, bias)) in enumerate(zip(L2_VALUES, stored_wb)):
        hidden_args = data_test['spins'] @ weights + bias
        ax = axs_flat[i]
        for j in range(hidden_args.shape[1]):
            ax.scatter(m_test, hidden_args[:, j], s=1, alpha=0.3,
                       color=plt.cm.jet(j / weights.shape[1]))
        ax.set_title(f'l2={fmt(l2)}', fontsize=9)
        ax.set_xlabel('magnetization', fontsize=8)
        ax.set_ylabel('h_j', fontsize=8)

    for i in range(n, len(axs_flat)):
        axs_flat[i].set_visible(False)

    plt.suptitle(f'Hidden args — L={L}  (L2 only)', fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'hidden_args_L{L}_all.png', dpi=120)
    plt.close()

    best_i      = int(np.nanargmin(composite))
    best_params[L] = L2_VALUES[best_i]
    print(f"→ Best  L={L}: l2={fmt(best_params[L])}"
          f"  composite={composite[best_i]:.5f}")


# ── Retrain best params (longer) and save best hidden-args panel ───────────────
print("\nRetraining best params (long run) for all L …")
fig, axs = plt.subplots(1, len(L_SIZES), figsize=(20, 4))

for i, L in enumerate(L_SIZES):
    best_l2 = best_params[L]
    print(f"  L={L}: l2={fmt(best_l2)}")

    train_conf, train_label, val_conf, val_label = load_split(L)
    best_model, _, _ = train(train_conf, train_label, val_conf, val_label,
                             train_conf.shape[1], best_l2, BEST_EPOCHS, BEST_PATIENCE)

    data_test     = np.load(f"{DATA_TEST_DIR}/L{L}_ising.npz")
    weights, bias = best_model.layers[1].get_weights()
    hidden_args   = data_test['spins'] @ weights + bias
    m             = np.mean(data_test['spins'], axis=1)

    for j in range(hidden_args.shape[1]):
        axs[i].scatter(m, hidden_args[:, j], s=1, alpha=0.3,
                       color=plt.cm.jet(j / weights.shape[1]))
    axs[i].set_title(f'L={L}  l2={fmt(best_l2)}')
    axs[i].set_xlabel('magnetization')
    axs[i].set_ylabel('argument of hidden neuron')

plt.tight_layout()
plt.savefig(OUT_DIR / 'hidden_args_best_v3.png', dpi=120)
plt.close()

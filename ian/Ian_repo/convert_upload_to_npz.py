import numpy as np
import os

# ── Temperature arrays ────────────────────────────────────────────────────────

# Square lattice temperatures (provided); Temp[20] = Tc_sq exactly
Temp_sq = np.array([
    1.0000000000000000, 1.0634592657106510, 1.1269185314213019, 1.1903777971319529,
    1.2538370628426039, 1.3172963285532548, 1.3807555942639058, 1.4442148599745568,
    1.5076741256852078, 1.5711333913958587, 1.6345926571065097, 1.6980519228171607,
    1.7615111885278116, 1.8249704542384626, 1.8884297199491136, 1.9518889856597645,
    2.0153482513704155, 2.0788075170810667, 2.1422667827917179, 2.2057260485023691,
    2.2691853142130203, 2.3326445799236715, 2.3961038456343227, 2.4595631113449739,
    2.5230223770556250, 2.5864816427662762, 2.6499409084769274, 2.7134001741875786,
    2.7768594398982298, 2.8403187056088810, 2.9037779713195322, 2.9672372370301834,
    3.0306965027408346, 3.0941557684514858, 3.1576150341621370, 3.2210742998727881,
    3.2845335655834393, 3.3479928312940905, 3.4114520970047417, 3.4749113627153929,
    3.5383706284260401
], dtype=np.float32)

# Triangular lattice: same step size, centred on Tc_tri
Tc_tri = np.float32(4 / np.log(3))
step   = float(Temp_sq[1] - Temp_sq[0])
Temp_tri = np.array([Tc_tri + (k - 20) * step for k in range(41)], dtype=np.float32)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_spins(path):
    """Load a whitespace-separated spin file (0/1) and convert to int8 (-1/+1)."""
    raw = np.loadtxt(path, dtype=np.int8)
    return (2 * raw - 1).astype(np.int8)   # 0 → -1, 1 → +1

def temps_train(Temp):
    """
    100 000-sample temperature vector for train files.
    Structure: 40 temperatures (Temp[0:20] + Temp[21:41]), 2500 samples each.
    Data is sorted: all y=0 (ordered) first, then all y=1 (disordered).
    Temp[20] = Tc is skipped (not used in training).
    """
    return np.concatenate([
        np.repeat(Temp[0:20],  2500),   # 50 000 ordered samples
        np.repeat(Temp[21:41], 2500),   # 50 000 disordered samples
    ])

def temps_test(Temp):
    """
    10 250-sample temperature vector for test files.
    Structure: 41 temperatures × 250 samples each.
    Tc (Temp[20]) appears in both phases: 125 samples labelled y=0 and 125 labelled y=1.
    """
    return np.concatenate([
        np.repeat(Temp[0:20],  250),   # 5 000 ordered samples
        np.repeat(Temp[20:21], 125),   # 125 ordered samples at Tc
        np.repeat(Temp[20:21], 125),   # 125 disordered samples at Tc
        np.repeat(Temp[21:41], 250),   # 5 000 disordered samples
    ])

# ── Paths ─────────────────────────────────────────────────────────────────────

UPLOAD_DIR  = os.path.join(os.path.dirname(__file__),
                            '..', '..', '..', 'to_upload')
UPLOAD_DIR  = os.path.abspath(UPLOAD_DIR)

OUT_ISING   = os.path.join(os.path.dirname(__file__), 'data_uploaded')
OUT_TRI     = os.path.join(os.path.dirname(__file__), 'tri_uploaded')

os.makedirs(OUT_ISING, exist_ok=True)
os.makedirs(OUT_TRI,   exist_ok=True)

# ── Convert ───────────────────────────────────────────────────────────────────

L_values = [10, 20, 30, 40, 60]

for L in L_values:
    folder = os.path.join(UPLOAD_DIR, f'L_{L}')
    print(f'\nProcessing L={L}...')

    # ── Ising ──────────────────────────────────────────────────────────────────
    spins_train = load_spins(os.path.join(folder, 'Xtrain.txt'))
    spins_test  = load_spins(os.path.join(folder, 'Xtest.txt'))

    T_train = temps_train(Temp_sq)
    T_test  = temps_test(Temp_sq)

    spins_ising = np.vstack([spins_train, spins_test])
    T_ising     = np.concatenate([T_train, T_test])

    out_path = os.path.join(OUT_ISING, f'L{L}_ising.npz')
    np.savez_compressed(out_path, temperatures=T_ising, spins=spins_ising)
    print(f'  Saved {out_path}  '
          f'(spins {spins_ising.shape}, temps {T_ising.shape})')

    # ── Triangular ─────────────────────────────────────────────────────────────
    spins_tri = load_spins(os.path.join(folder, 'TRIANGXtest.txt'))
    T_tri_arr = temps_test(Temp_tri)   # same 10 250-sample structure

    out_path_tri = os.path.join(OUT_TRI, f'L{L}_tri.npz')
    np.savez_compressed(out_path_tri, temperatures=T_tri_arr, spins=spins_tri)
    print(f'  Saved {out_path_tri}  '
          f'(spins {spins_tri.shape}, temps {T_tri_arr.shape})')

print('\nDone.')

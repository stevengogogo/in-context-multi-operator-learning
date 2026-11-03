"""Generate and save the shared 1D Poisson dataset."""
import os, sys, time
from pathlib import Path as _P
_HERE = _P(__file__).resolve().parent
_DATA = str(_HERE.parents[1] / "data" / "icon_comparison.npz")
_RES = str(_HERE.parent / "output" / "icon_comparison") + "/"
os.makedirs(_RES, exist_ok=True)
import numpy as np

N_OPS    = int(os.environ.get("N_OPS",    "5000"))
N_PAIRS  = int(os.environ.get("N_PAIRS",  "40"))
N        = int(os.environ.get("N_GRID",   "50"))
SEED     = int(os.environ.get("SEED",     "42"))
OUT_FILE = os.environ.get("DATA_FILE",    _DATA)

def solve_poisson1d_fd(k, f, N):
    dx = 1.0 / (N + 1)
    k_half_p = 0.5 * (k[1:-1] + k[2:])
    k_half_m = 0.5 * (k[:-2]  + k[1:-1])
    diag  =  (k_half_p + k_half_m) / dx**2
    upper = -k_half_p[:-1] / dx**2
    lower = -k_half_m[1:]  / dx**2
    n = N
    c_ = np.zeros(n); d_ = f.copy().astype(float)
    c_[0] = upper[0] / diag[0]; d_[0] = d_[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i-1]*c_[i-1]
        if i < n-1:
            c_[i] = upper[i] / denom
        d_[i] = (d_[i] - lower[i-1]*d_[i-1]) / denom
    u = np.zeros(n); u[-1] = d_[-1]
    for i in range(n-2, -1, -1):
        u[i] = d_[i] - c_[i]*u[i+1]
    return u

def rbf_kernel(x, sigma, l):
    diff = x[:, None] - x[None, :]
    return sigma**2 * np.exp(-0.5 * diff**2 / l**2)

def sample_gp(rng, x, n_samples, sigma, l):
    K = rbf_kernel(x, sigma, l) + 1e-6 * np.eye(len(x))
    L = np.linalg.cholesky(K)
    return (L @ rng.standard_normal((len(x), n_samples))).T

rng = np.random.default_rng(SEED)
xs_full = np.linspace(0, 1, N + 2)
xs_int  = xs_full[1:-1]
k = np.exp(sample_gp(rng, xs_full, N_OPS, 0.5, 0.4))
f = sample_gp(rng, xs_int, N_OPS * N_PAIRS, 2.0, 0.4).reshape(N_OPS, N_PAIRS, N)
u = np.zeros_like(f)
for i in range(N_OPS):
    for j in range(N_PAIRS):
        u[i, j] = solve_poisson1d_fd(k[i], f[i, j], N)

np.savez(OUT_FILE, f=f, u=u, xs=xs_int, N_OPS=N_OPS, N_PAIRS=N_PAIRS, N=N, SEED=SEED)
print(f"Saved {OUT_FILE}  f:{f.shape}  u:{u.shape}", flush=True)

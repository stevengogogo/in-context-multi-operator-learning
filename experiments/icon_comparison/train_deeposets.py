"""DeepOSets worker — runs in its own process with a clean GPU context."""
import os, time, json
from pathlib import Path as _P
_HERE = _P(__file__).resolve().parent
_DATA = str(_HERE.parents[1] / "data" / "icon_comparison.npz")
_RES = str(_HERE.parent / "output" / "icon_comparison") + "/"
os.makedirs(_RES, exist_ok=True)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_FILE  = os.environ.get("DATA_FILE", _DATA)
OUT_FILE   = os.environ.get("OUT_FILE",    _RES + "deeposets_results.json")
N_CONTEXT  = int(os.environ.get("N_CONTEXT",  "5"))
N_STEPS    = int(os.environ.get("N_STEPS",    "100000"))
WIDTH      = int(os.environ.get("WIDTH",      "256"))
BATCH      = int(os.environ.get("BATCH",      "5"))
SEED       = int(os.environ.get("SEED",       "42"))
TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC","0.9"))

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.load(DATA_FILE)
f_all, u_all, xs = data["f"], data["u"], data["xs"]
N = xs.shape[0]
n_train = int(len(f_all) * TRAIN_FRAC)
f_train, u_train = f_all[:n_train], u_all[:n_train]
f_test,  u_test  = f_all[n_train:], u_all[n_train:]
n_pairs = f_train.shape[1]

# Pre-convert training data to GPU tensors
f_train_t = torch.tensor(f_train, dtype=torch.float32, device=device)
u_train_t = torch.tensor(u_train, dtype=torch.float32, device=device)

# xs_t: (N, 1) query coordinates on device
xs_t = torch.tensor(xs[:, None], dtype=torch.float32, device=device)

# ── model ─────────────────────────────────────────────────────────────────────
def make_mlp(in_size, out_size, widths):
    sizes = [in_size] + widths + [out_size]
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(a, b))
        if b != out_size:
            layers.append(nn.ReLU())
    # replace the last ReLU-less append: build manually
    return nn.Sequential(*layers)

def _build_mlp(in_size, out_size, widths):
    sizes = [in_size] + widths + [out_size]
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class DeepOSets(nn.Module):
    def __init__(self, n, width):
        super().__init__()
        self.phi    = _build_mlp(2 * n, width, [width] * 4)
        self.branch = _build_mlp(n + width, width, [width] * 4)
        self.trunk  = _build_mlp(1, width, [width] * 4)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, cond_ctx, qoi_ctx, cond_q, xs_q):
        # cond_ctx: (B, m, N)  qoi_ctx: (B, m, N)
        # cond_q:   (B, N)     xs_q:    (Ng, 1)
        B, m, Ng = cond_ctx.shape
        pairs = torch.cat([cond_ctx, qoi_ctx], dim=-1)          # (B, m, 2N)
        h = self.phi(pairs).mean(dim=1)                         # (B, width)
        branch_in = torch.cat([cond_q, h], dim=-1)              # (B, N+width)
        b = self.branch(branch_in)                              # (B, width)
        t = self.trunk(xs_q)                                    # (Ng, width)
        # dot product over width for each query point: (B, Ng)
        out = (b.unsqueeze(1) * t.unsqueeze(0)).sum(-1) + self.bias
        return out                                              # (B, Ng)

model = DeepOSets(N, WIDTH).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"[DeepOSets] params={n_params:,}  device={device}", flush=True)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)

# ── training ──────────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
t0 = time.perf_counter()
model.train()
for i in range(N_STEPS):
    op_idx  = rng.integers(0, len(f_train), size=BATCH)
    perm    = [rng.permutation(n_pairs) for _ in range(BATCH)]
    ctx_idx = np.stack([p[:N_CONTEXT] for p in perm])
    qry_idx = np.array([p[N_CONTEXT]  for p in perm])

    op_t  = torch.tensor(op_idx,  dtype=torch.long)
    ctx_t = torch.tensor(ctx_idx, dtype=torch.long)
    qry_t = torch.tensor(qry_idx, dtype=torch.long)

    cf = f_train_t[op_t[:, None], ctx_t]   # (B, N_CONTEXT, N)
    cq = u_train_t[op_t[:, None], ctx_t]   # (B, N_CONTEXT, N)
    fq = f_train_t[op_t, qry_t]            # (B, N)
    uq = u_train_t[op_t, qry_t]            # (B, N)

    opt.zero_grad()
    pred = model(cf, cq, fq, xs_t)
    loss = F.mse_loss(pred, uq)
    loss.backward()
    opt.step()

    if (i + 1) % 10_000 == 0:
        print(f"  step {i+1}/{N_STEPS}  loss={loss.item():.4e}", flush=True)

train_time = time.perf_counter() - t0
print(f"[DeepOSets] train_time={train_time:.1f}s", flush=True)

# ── evaluation ────────────────────────────────────────────────────────────────
model.eval()
# Warmup
with torch.no_grad():
    _rng_w = np.random.default_rng(0)
    _op_w  = _rng_w.integers(0, len(f_test))
    _perm_w = _rng_w.permutation(f_test.shape[1])
    _cf_w = torch.tensor(f_test[_op_w, _perm_w[:N_CONTEXT]][None], dtype=torch.float32, device=device)
    _cq_w = torch.tensor(u_test[_op_w, _perm_w[:N_CONTEXT]][None], dtype=torch.float32, device=device)
    _fq_w = torch.tensor(f_test[_op_w, _perm_w[N_CONTEXT]][None],  dtype=torch.float32, device=device)
    if device.type == "cuda": torch.cuda.synchronize()
    model(_cf_w, _cq_w, _fq_w, xs_t)
    if device.type == "cuda": torch.cuda.synchronize()
print("[DeepOSets] warmup done", flush=True)

rel_l2 = {}; infer_times = []
with torch.no_grad():
    for n_ctx in range(1, N_CONTEXT + 1):
        errs = []
        rng2 = np.random.default_rng(SEED + 1)
        for _ in range(100):
            op_i = rng2.integers(0, len(f_test))
            perm = rng2.permutation(f_test.shape[1])
            ci, qi = perm[:n_ctx], perm[n_ctx]
            cf = torch.tensor(f_test[op_i, ci][None], dtype=torch.float32, device=device)
            cq = torch.tensor(u_test[op_i, ci][None], dtype=torch.float32, device=device)
            fq = torch.tensor(f_test[op_i, qi][None], dtype=torch.float32, device=device)
            uq = u_test[op_i, qi]
            # pad context to N_CONTEXT if n_ctx < N_CONTEXT
            pad = N_CONTEXT - n_ctx
            if pad > 0:
                cf = torch.cat([cf, torch.zeros(1, pad, N, device=device)], dim=1)
                cq = torch.cat([cq, torch.zeros(1, pad, N, device=device)], dim=1)
            if device.type == "cuda": torch.cuda.synchronize()
            t_i = time.perf_counter()
            out = model(cf, cq, fq, xs_t)
            if device.type == "cuda": torch.cuda.synchronize()
            infer_times.append(time.perf_counter() - t_i)
            pred_np = out[0].cpu().numpy()
            errs.append(np.linalg.norm(pred_np - uq) / (np.linalg.norm(uq) + 1e-8))
        rel_l2[n_ctx] = float(np.mean(errs))
        print(f"  n_ctx={n_ctx}  rel-L2={rel_l2[n_ctx]:.4f}", flush=True)

infer_ms = float(np.mean(infer_times)) * 1e3
print(f"[DeepOSets] infer_ms={infer_ms:.3f}", flush=True)

with open(OUT_FILE, "w") as f_out:
    json.dump({"n_params": n_params, "train_time": train_time,
               "steps_done": N_STEPS,
               "infer_ms": infer_ms, "rel_l2": {str(k): v for k, v in rel_l2.items()}}, f_out)
print(f"Saved {OUT_FILE}", flush=True)

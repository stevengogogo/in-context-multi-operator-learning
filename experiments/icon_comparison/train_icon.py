"""ICON worker — runs in its own process with a clean GPU context."""
import os, sys, time, json
from pathlib import Path as _P
_HERE = _P(__file__).resolve().parent
_DATA = str(_HERE.parents[1] / "data" / "icon_comparison.npz")
_RES = str(_HERE.parent / "output" / "icon_comparison") + "/"
os.makedirs(_RES, exist_ok=True)
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DATA_FILE      = os.environ.get("DATA_FILE", _DATA)
OUT_FILE       = os.environ.get("OUT_FILE",       _RES + "icon_results.json")
N_CONTEXT      = int(os.environ.get("N_CONTEXT",     "5"))
N_STEPS        = int(os.environ.get("N_STEPS",       "100000"))
TIME_BUDGET_S  = float(os.environ.get("TIME_BUDGET_S", "0"))  # 0 = use N_STEPS
MODEL_DIM      = int(os.environ.get("MODEL_DIM",     "256"))
N_LAYERS       = int(os.environ.get("N_LAYERS",      "6"))
N_HEADS        = int(os.environ.get("N_HEADS",       "8"))
WIDENING       = int(os.environ.get("WIDENING",      "4"))
BATCH          = int(os.environ.get("BATCH",         "32"))
SEED           = int(os.environ.get("SEED",          "42"))
TRAIN_FRAC     = float(os.environ.get("TRAIN_FRAC",  "0.9"))

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = np.load(DATA_FILE)
f_all, u_all, xs = data["f"], data["u"], data["xs"]
N = xs.shape[0]
n_train = int(len(f_all) * TRAIN_FRAC)
f_train, u_train = f_all[:n_train], u_all[:n_train]
f_test,  u_test  = f_all[n_train:], u_all[n_train:]
n_pairs = f_train.shape[1]

# Pre-convert full training arrays to GPU tensors once
f_train_t = torch.tensor(f_train, dtype=torch.float32, device=device)
u_train_t = torch.tensor(u_train, dtype=torch.float32, device=device)

# Pre-cached sequence constants (fixed n_demo = N_CONTEXT during training)
SEQ_LEN = (N_CONTEXT * 3 + 2) * N
_pos_ids = []
for j in range(N_CONTEXT):
    _pos_ids += [j*3]*N + [j*3+1]*N + [j*3+2]*N
_pos_ids += [N_CONTEXT*3]*N + [N_CONTEXT*3+2]*N
pos_ids = torch.tensor(_pos_ids, dtype=torch.long, device=device)

xgrid_rep = torch.tensor(
    np.concatenate([np.linspace(0,1,N)]*(N_CONTEXT*3+2)),
    dtype=torch.float32, device=device)

class ICONPoisson(nn.Module):
    def __init__(self):
        super().__init__()
        max_pos = N_CONTEXT * 3 + 3
        self.input_proj = nn.Linear(2, MODEL_DIM)
        self.pos_embed  = nn.Embedding(max_pos, MODEL_DIM)
        enc = nn.TransformerEncoderLayer(MODEL_DIM, N_HEADS,
            MODEL_DIM*WIDENING, dropout=0.0, activation='gelu',
            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=N_LAYERS)
        self.output_proj = nn.Linear(MODEL_DIM, 1)

    def forward(self, values):
        B = values.shape[0]
        tokens = torch.stack([xgrid_rep.expand(B,-1), values], dim=-1)
        emb = self.input_proj(tokens) + self.pos_embed(pos_ids).unsqueeze(0)
        out = self.transformer(emb)
        return self.output_proj(out[:, -N:, :]).squeeze(-1)

def build_values(demo_f, demo_u, quest_f):
    B, m, Ng = demo_f.shape
    parts = []
    for j in range(m):
        parts += [demo_f[:,j,:], demo_u[:,j,:], torch.zeros(B,Ng,device=demo_f.device)]
    parts += [quest_f, torch.zeros(B,Ng,device=demo_f.device)]
    return torch.cat(parts, dim=1)

model = ICONPoisson().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"[ICON] params={n_params:,}  device={device}  seq_len={SEQ_LEN}", flush=True)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
rng = np.random.default_rng(SEED)

t0 = time.perf_counter()
model.train()
steps_done = 0
max_steps = N_STEPS #if TIME_BUDGET_S <= 0 else 10_000_000  # effectively unlimited
for i in range(max_steps):
    if TIME_BUDGET_S > 0 and (time.perf_counter() - t0) >= TIME_BUDGET_S:
        break
    op_idx   = rng.integers(0, len(f_train), size=BATCH)
    perms    = [rng.permutation(n_pairs) for _ in range(BATCH)]
    demo_idx = np.stack([p[:N_CONTEXT] for p in perms])
    qry_idx  = np.array([p[N_CONTEXT]  for p in perms])
    op_t  = torch.tensor(op_idx, dtype=torch.long)
    dem_t = torch.tensor(demo_idx, dtype=torch.long)
    qry_t = torch.tensor(qry_idx, dtype=torch.long)
    df = f_train_t[op_t[:, None], dem_t]   # (B, N_CONTEXT, N) — no CPU→GPU copy
    du = u_train_t[op_t[:, None], dem_t]
    qf = f_train_t[op_t, qry_t]
    qu = u_train_t[op_t, qry_t]
    opt.zero_grad()
    loss = F.mse_loss(model(build_values(df, du, qf)), qu)
    loss.backward(); opt.step()
    steps_done += 1
    if steps_done % 1_000 == 0:
        elapsed = time.perf_counter() - t0
        budget_str = f"/{TIME_BUDGET_S:.0f}s" if TIME_BUDGET_S > 0 else f"/{N_STEPS}"
        print(f"  step {steps_done}{budget_str}  loss={loss.item():.4e}  elapsed={elapsed:.0f}s  ~{elapsed/steps_done*(N_STEPS-steps_done)/60:.1f}min left", flush=True)

train_time = time.perf_counter() - t0
print(f"[ICON] train_time={train_time:.1f}s  steps={steps_done}", flush=True)

model.eval()
rel_l2 = {}; infer_times = []
with torch.no_grad():
    for n_ctx in range(1, N_CONTEXT+1):
        errs = []
        rng2 = np.random.default_rng(SEED+1)
        for _ in range(100):
            op_i = rng2.integers(0, len(f_test))
            perm = rng2.permutation(f_test.shape[1])
            ci, qi = perm[:n_ctx], perm[n_ctx]
            df = torch.tensor(f_test[op_i,ci][None], dtype=torch.float32).to(device)
            du = torch.tensor(u_test[op_i,ci][None], dtype=torch.float32).to(device)
            qf = torch.tensor(f_test[op_i,qi][None], dtype=torch.float32).to(device)
            qu_true = u_test[op_i, qi]
            pad = N_CONTEXT - n_ctx
            if pad > 0:
                df = torch.cat([df, torch.zeros(1,pad,N,device=device)], 1)
                du = torch.cat([du, torch.zeros(1,pad,N,device=device)], 1)
            if device.type=='cuda': torch.cuda.synchronize()
            t_i = time.perf_counter()
            pred = model(build_values(df, du, qf))[0]
            if device.type=='cuda': torch.cuda.synchronize()
            infer_times.append(time.perf_counter()-t_i)
            errs.append(np.linalg.norm(pred.cpu().numpy()-qu_true)/(np.linalg.norm(qu_true)+1e-8))
        rel_l2[n_ctx] = float(np.mean(errs))
        print(f"  n_ctx={n_ctx}  rel-L2={rel_l2[n_ctx]:.4f}", flush=True)

infer_ms = float(np.mean(infer_times))*1e3
print(f"[ICON] infer_ms={infer_ms:.3f}", flush=True)
with open(OUT_FILE,"w") as fout:
    json.dump({"n_params":n_params,"train_time":train_time,
               "steps_done": steps_done,
               "infer_ms":infer_ms,"rel_l2":{str(k):v for k,v in rel_l2.items()}},fout)
print(f"Saved {OUT_FILE}", flush=True)

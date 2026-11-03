"""
Benchmark: DeepOSets vs ICON for 1D Poisson
============================================
Orchestrator – runs data generation, DeepOSets, and ICON each in a *separate*
subprocess so JAX and PyTorch never share GPU memory (fair comparison).

Usage:
    python experiments/icon_comparison/run_comparison.py
"""

import os, sys, subprocess, json, time
from pathlib import Path as _P
_HERE = _P(__file__).resolve().parent
_DATA = str(_HERE.parents[1] / "data" / "icon_comparison.npz")
_RES = str(_HERE.parent / "output" / "icon_comparison") + "/"
os.makedirs(_RES, exist_ok=True)
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────
N_OPS      = 5_000
N_PAIRS    = 40
N_GRID     = 50
N_CONTEXT  = 5
N_STEPS    = 100_000
SEED       = 42
TRAIN_FRAC = 0.9

DATA_FILE       = _DATA
DOS_RESULT_FILE = _RES + "deeposets_results.json"
ICON_RESULT_FILE= _RES + "icon_results.json"
MD_FILE         = _RES + "benchmark_results.md"

PYTHON = sys.executable   # same interpreter / venv

def run(script, extra_env=None):
    env = {**os.environ,
           "DATA_FILE":   DATA_FILE,
           "N_CONTEXT":   str(N_CONTEXT),
           "N_STEPS":     str(N_STEPS),
           "SEED":        str(SEED),
           "TRAIN_FRAC":  str(TRAIN_FRAC),
           **(extra_env or {})}
    proc = subprocess.run([PYTHON, str(_HERE / script)], env=env)
    if proc.returncode != 0:
        print(f"ERROR: {script} exited with code {proc.returncode}", flush=True)
        sys.exit(proc.returncode)

# ── 1. generate data ──────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1 / 3 — Generating 1D Poisson dataset …")
t0 = time.perf_counter()
run("datagen.py", {"N_OPS": str(N_OPS), "N_PAIRS": str(N_PAIRS),
                         "N_GRID": str(N_GRID), "DATA_FILE": DATA_FILE})
print(f"  done in {time.perf_counter()-t0:.1f}s\n")

# ── 2. DeepOSets (clean GPU context) ─────────────────────────────────────────
print("=" * 60)
print("Step 2 / 3 — Training DeepOSets (JAX, isolated process) …")
run("train_deeposets.py", {"OUT_FILE": DOS_RESULT_FILE, "WIDTH": "256", "BATCH": "5"})

# ── 3. ICON (clean GPU context, same step budget as DeepOSets) ─────────────
print("\n" + "=" * 60)
print(f"Step 3 / 3 — Training ICON (PyTorch, isolated process, {N_STEPS:,} steps) …")
run("train_icon.py", {"OUT_FILE": ICON_RESULT_FILE,
                      "MODEL_DIM": "256", "N_LAYERS": "6",
                      "N_HEADS": "8", "WIDENING": "4", "BATCH": "32",
                      "N_STEPS": str(N_STEPS), "TIME_BUDGET_S": "0"})

# ── 4. Collect results and write markdown ─────────────────────────────────────
with open(DOS_RESULT_FILE)  as f: dos  = json.load(f)
with open(ICON_RESULT_FILE) as f: icon = json.load(f)

dos_l2  = {int(k): v for k, v in dos["rel_l2"].items()}
icon_l2 = {int(k): v for k, v in icon["rel_l2"].items()}

print("\n" + "=" * 60)
print("BENCHMARK RESULTS — 1D Poisson  (f → u)")
print("=" * 60)
print(f"{'Metric':<42} {'DeepOSets':>14} {'ICON':>14}")
print("-" * 72)
print(f"{'Trainable parameters':<42} {dos['n_params']:>14,} {icon['n_params']:>14,}")
print(f"{'Training time (s)':<42} {dos['train_time']:>14.1f} {icon['train_time']:>14.1f}")
print(f"{'Training steps completed':<42} {dos.get('steps_done', N_STEPS):>14,} {icon.get('steps_done', '?'):>14}")
print(f"{'Inference time / sample (ms)':<42} {dos['infer_ms']:>14.3f} {icon['infer_ms']:>14.3f}")
print("-" * 72)
for n in range(1, N_CONTEXT + 1):
    dv = dos_l2.get(n, float('nan'))
    iv = icon_l2.get(n, float('nan'))
    print(f"  Test rel-L2  n_ctx={n}{'':29} {dv:>14.4f} {iv:>14.4f}")

md = [
    "# Benchmark: DeepOSets vs ICON — 1D Poisson (`f → u`)",
    "",
    "> Each model was trained in an **isolated subprocess** so JAX and PyTorch",
    "> never share GPU memory (fair comparison).",
    "",
    "## Setup",
    "",
    "| Setting | Value |",
    "|---------|-------|",
    f"| Grid points N | {N_GRID} |",
    f"| Operators (total) | {N_OPS:,} |",
    f"| Pairs per operator | {N_PAIRS} |",
    f"| Train / test split | {int(TRAIN_FRAC*100)}% / {100-int(TRAIN_FRAC*100)}% |",
    f"| Training steps | {N_STEPS:,} |",
    f"| DeepOSets width | 256 |",
    f"| ICON model_dim / layers / heads | 256 / 6 / 8 |",
    f"| Hardware | NVIDIA RTX A2000 Laptop GPU |",
    "",
    "## Results",
    "",
    "| Metric | DeepOSets | ICON |",
    "|--------|----------:|-----:|",
    f"| Trainable parameters | {dos['n_params']:,} | {icon['n_params']:,} |",
    f"| Training time (s) | {dos['train_time']:.1f} | {icon['train_time']:.1f} |",
    f"| Inference time / sample (ms) | {dos['infer_ms']:.3f} | {icon['infer_ms']:.3f} |",
]
for n in range(1, N_CONTEXT + 1):
    md.append(f"| Test rel-L2 (n_ctx={n}) | {dos_l2.get(n,float('nan')):.4f} | {icon_l2.get(n,float('nan')):.4f} |")

md_text = "\n".join(md) + "\n"
with open(MD_FILE, "w") as f:
    f.write(md_text)

print(f"\nMarkdown results saved to {MD_FILE}")
print(md_text)


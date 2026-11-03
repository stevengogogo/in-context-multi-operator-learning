"""Nearest-neighbor operator-lookup baseline for the 1D Poisson/Darcy-flow
in-context operator-learning task.

Rebuttal context
----------------
A reviewer raised the concern that, given the size of the training dataset,
the task may be "trivial" in the sense that a learned in-context model such
as DeepOSets or ICON may perform no better than a simple nearest-neighbor
lookup over the training pool. This script implements exactly such a
baseline, using *no learned parameters at all*:

  1. Given the n in-context (condition, solution) pairs describing an unseen
     test operator, find the single training operator whose own
     demonstration pairs are jointly closest (in condition AND solution) to
     the given context. This is the "nearest-neighbor training example".
  2. Predict the query solution by looking up, *within that training
     operator's own pairs*, the one whose condition is closest to the query
     condition, and returning its stored solution.

The entire training pool (all training operators x all pairs) is used
purely as a lookup table; no gradient-based training occurs.

Usable both as a standalone script (writes a JSON results file in the same
format as train_deeposets.py / train_icon.py) and as a library
(`run_nn_baseline`) for the dataset-size ablation in
dataset_ablation.py.
"""
import os, time, json
from pathlib import Path as _P
_HERE = _P(__file__).resolve().parent
_DATA = str(_HERE.parents[1] / "data" / "icon_comparison.npz")
_RES = str(_HERE.parent / "output" / "icon_comparison") + "/"
os.makedirs(_RES, exist_ok=True)
import numpy as np
import torch

DATA_FILE      = os.environ.get("DATA_FILE",      _DATA)
OUT_FILE       = os.environ.get("OUT_FILE",       _RES + "nn_baseline_results.json")
N_CONTEXT      = int(os.environ.get("N_CONTEXT",      "5"))
SEED           = int(os.environ.get("SEED",           "42"))
TRAIN_FRAC     = float(os.environ.get("TRAIN_FRAC",   "0.9"))
N_TRAIN_SUBSET = int(os.environ.get("N_TRAIN_SUBSET", "-1"))   # -1 = use all training operators
N_QUERIES      = int(os.environ.get("N_QUERIES",      "100"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_nn_baseline(f_train, u_train, f_test, u_test, n_context, seed,
                     n_queries=100, device=device):
    """Operator-level nearest-neighbor baseline.

    f_train, u_train : (K, P, N) training pool (K operators, P pairs, N grid pts)
    f_test,  u_test  : (M, P, N) held-out test operators
    Returns a dict with per-context rel-L2 error, mean inference time (ms),
    and the size of the training pool used.
    """
    K, P, N = f_train.shape
    f_train_t = torch.tensor(f_train, dtype=torch.float32, device=device)
    u_train_t = torch.tensor(u_train, dtype=torch.float32, device=device)
    # Candidate pairs, flattened over (K, P): joint (f, u) vectors used for
    # matching against the given in-context demonstrations.
    cand_fu = torch.cat([f_train_t, u_train_t], dim=-1).reshape(K * P, 2 * N)

    rel_l2 = {}
    infer_times = []
    for n_ctx in range(1, n_context + 1):
        errs = []
        rng2 = np.random.default_rng(seed + 1)
        for _ in range(n_queries):
            op_i = rng2.integers(0, len(f_test))
            perm = rng2.permutation(f_test.shape[1])
            ci, qi = perm[:n_ctx], perm[n_ctx]
            f_ctx = torch.tensor(f_test[op_i, ci], dtype=torch.float32, device=device)  # (n_ctx, N)
            u_ctx = torch.tensor(u_test[op_i, ci], dtype=torch.float32, device=device)  # (n_ctx, N)
            f_q = f_test[op_i, qi]
            u_q = u_test[op_i, qi]

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # 1) Find the nearest-neighbor training operator: for each
            #    context example, the closest candidate pair (jointly in
            #    f and u), summed over context examples per candidate op.
            ctx_fu = torch.cat([f_ctx, u_ctx], dim=-1)                       # (n_ctx, 2N)
            d = torch.cdist(ctx_fu, cand_fu).pow(2).reshape(n_ctx, K, P)     # (n_ctx, K, P)
            best_per_op, _ = d.min(dim=-1)                                  # (n_ctx, K)
            total_per_op = best_per_op.sum(dim=0)                           # (K,)
            c_star = int(torch.argmin(total_per_op).item())

            # 2) Within that operator, look up the pair whose condition is
            #    nearest to the query condition, and output its solution.
            f_q_t = torch.tensor(f_q, dtype=torch.float32, device=device)
            d_q = torch.cdist(f_q_t[None, :], f_train_t[c_star]).squeeze(0)  # (P,)
            p_star = int(torch.argmin(d_q).item())
            pred = u_train_t[c_star, p_star].cpu().numpy()

            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_times.append(time.perf_counter() - t0)

            errs.append(float(np.linalg.norm(pred - u_q) / (np.linalg.norm(u_q) + 1e-8)))
        rel_l2[n_ctx] = float(np.mean(errs))

    infer_ms = float(np.mean(infer_times)) * 1e3
    return {"rel_l2": rel_l2, "infer_ms": infer_ms, "train_pool_size": int(K)}


if __name__ == "__main__":
    data = np.load(DATA_FILE)
    f_all, u_all, xs = data["f"], data["u"], data["xs"]
    n_train_full = int(len(f_all) * TRAIN_FRAC)
    f_train_full, u_train_full = f_all[:n_train_full], u_all[:n_train_full]
    f_test, u_test = f_all[n_train_full:], u_all[n_train_full:]

    K = N_TRAIN_SUBSET if N_TRAIN_SUBSET > 0 else n_train_full
    K = min(K, n_train_full)
    f_train, u_train = f_train_full[:K], u_train_full[:K]

    print(f"[NN-baseline] train_pool_size={K}  test_ops={len(f_test)}  device={device}", flush=True)
    t0 = time.perf_counter()
    result = run_nn_baseline(f_train, u_train, f_test, u_test, N_CONTEXT, SEED,
                              n_queries=N_QUERIES, device=device)
    wall = time.perf_counter() - t0
    for n_ctx, v in result["rel_l2"].items():
        print(f"  n_ctx={n_ctx}  rel-L2={v:.4f}", flush=True)
    print(f"[NN-baseline] infer_ms={result['infer_ms']:.3f}  wall_time={wall:.1f}s", flush=True)

    out = {"n_params": 0, "train_time": 0.0, "steps_done": 0,
           "infer_ms": result["infer_ms"],
           "rel_l2": {str(k): v for k, v in result["rel_l2"].items()},
           "train_pool_size": K}
    with open(OUT_FILE, "w") as f_out:
        json.dump(out, f_out)
    print(f"Saved {OUT_FILE}", flush=True)

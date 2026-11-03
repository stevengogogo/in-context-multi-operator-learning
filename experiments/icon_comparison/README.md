# DeepOSets vs ICON on 1D Darcy / Poisson (Sec. 6.6, App. B.5, App. D)

Self-contained PyTorch scripts (CUDA optional). Configuration is via environment variables (`N_STEPS`, `N_CONTEXT`, `BATCH`, `SEED`, ...); defaults match the paper (5000 operators, 40 pairs, 50 grid points, 90/10 split, seed 42, Adam, MSE).

| Script | Purpose |
|--------|---------|
| `datagen.py` | Generates the dataset to `data/icon_comparison.npz` |
| `train_deeposets.py` | Trains/evaluates DeepOSets (4x256 MLPs, batch 5) |
| `train_icon.py` | Trains/evaluates ICON (6-layer Transformer, d=256, batch 32) |
| `run_comparison.py` | Orchestrator: runs the three above in separate processes and writes `experiments/output/icon_comparison/benchmark_results.md` |
| `nn_baseline.py` | Nearest-neighbor operator-lookup baseline (App. D) |
| `dataset_ablation.py` | Training-set-size ablation (App. D, Table 7) |

Run from the repository root, e.g. `python experiments/icon_comparison/run_comparison.py`. Results are written to `experiments/output/icon_comparison/` (git-ignored); timings depend on the GPU.

## Protocol notes

Both models use the same data, split, optimizer (Adam, lr 1e-4), loss (MSE), step budget and test samples. Differences to keep in mind when reading the comparison:

* Batch size is 5 operators for DeepOSets and 32 for ICON, so equal steps is not equal compute or equal data seen.
* ICON has about 5.3x the parameters of DeepOSets.
* Both models are trained with 5 context pairs. For fewer pairs at test time the context is zero-padded for both models, which can hurt DeepOSets at small context sizes.
* ICON here is a plain 6-layer Transformer, not the ICON-LM code in `third_party/`.
* One seed and 100 test samples per context size; latency is measured at batch size 1.

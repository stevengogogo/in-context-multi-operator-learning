# Experiments

How each paper experiment maps to code, how its data is generated, and how our reruns compare with the paper. Commands run from the repository root, prefixed with `uv run` (e.g. `uv run python scripts/train_1d.py`). Generated data goes to `data/` and results to `experiments/output/` (both git-ignored). `bash scripts/run_all.sh` runs everything.

## Status

| Paper | Experiment | Code | Status |
|---|---|---|---|
| 6.1 | Burgers | `scripts/*burgers*`, `train_1d.py --problem burgers` | Runnable |
| 6.2 | 1D Darcy | `scripts/*1d_poisson*`, `train_1d.py --problem poisson` | Runnable |
| 6.3 | 2D Darcy | `scripts/*2d*` | Runnable |
| 6.4 | 12 forward/inverse tasks | solvers in `deeposets/solvers/one_d.py` | No experiment script yet |
| 6.5 | Extrapolation to larger context | `evaluate_1d.py --min-ucpair/--max-ucpair` | Partial |
| 6.6, B.5 | DeepOSets vs ICON | `experiments/icon_comparison/` | Runnable |
| App. C | Prompt ablations | not included | Missing |
| App. D | Nearest-neighbor baseline | `experiments/icon_comparison/nn_baseline.py`, `dataset_ablation.py` | Runnable |

## Data generation

All data is synthetic: sample random input fields, then solve the PDE numerically. Seeds are set with `--seed` (default 0; 42 for the ICON comparison).

| Problem | Inputs | Solver | Code |
|---|---|---|---|
| Burgers | ν log-uniform in [0.001, 0.1]; initial condition from a GP (RBF σ=0.8, ℓ=0.2) times sin²(πx) | IMEX finite differences, 100 x-points, 1000 time steps | `generate_burgers1d_dataset` in [deeposets/datasets.py](../deeposets/datasets.py) |
| 1D Darcy | k = exp(GP, σ=0.5, ℓ=0.4); f ~ GP (σ=2, ℓ=0.4); 50 points | Second-order FD with arithmetic-average interface k, Thomas algorithm | `generate_poisson1d_dataset` |
| 2D Darcy | k = exp(random field, σ=0.3, ℓ=0.4); f random field (σ=5, ℓ=0.4), random Fourier features | FEM, bilinear quadrilateral elements | `generate_poisson2d_dataset`, [deeposets/solvers/two_d.py](../deeposets/solvers/two_d.py) |
| ICON comparison | As 1D Darcy, 5000 operators x 40 pairs | Thomas algorithm (NumPy) | [datagen.py](../experiments/icon_comparison/datagen.py) |

Each operator has several (input, solution) pairs. Datasets are split 90/10 by operator. During training, `num_ucpair - 1` pairs of one operator form the prompt and one more pair is the query.

## Commands

### Burgers (6.1)

```bash
python scripts/generate_1d_burgers.py
python scripts/train_1d.py --problem burgers
python scripts/evaluate_1d.py --problem burgers --min-ucpair 3 --max-ucpair 39 --n-trials 100 --n-eqns 100
python scripts/plot_1d_prediction.py --problem burgers
```

### 1D Darcy (6.2)

```bash
python scripts/generate_1d_poisson.py
python scripts/train_1d.py --problem poisson
python scripts/evaluate_1d.py --problem poisson --min-ucpair 3 --max-ucpair 39 --n-trials 100 --n-eqns 100
python scripts/plot_1d_prediction.py --problem poisson
```

### 2D Darcy (6.3)

```bash
python scripts/generate_2d_poisson.py
python scripts/train_2d.py
python scripts/evaluate_2d.py
python scripts/plot_2d_prediction.py
```

### Context-size extrapolation (6.5)

Train with a small prompt, then evaluate larger ones: `evaluate_1d.py --min-ucpair 3 --max-ucpair 20`. The model is permutation invariant and linear in prompt length, so any prompt size is valid at test time.

### DeepOSets vs ICON and nearest-neighbor baseline (6.6, B.5, D)

```bash
uv sync --extra icon
python experiments/icon_comparison/run_comparison.py
python experiments/icon_comparison/nn_baseline.py
python experiments/icon_comparison/dataset_ablation.py
```

See [experiments/icon_comparison/README.md](../experiments/icon_comparison/README.md).

## Paper numbers and this code

The Burgers, 1D Darcy and 2D Darcy results in the paper (relative L2 errors, final training losses, learning rate and training times) come from running this code with the defaults, at paper scale (200k operators for Burgers and 1D Darcy, 50k for 2D Darcy, 40 pairs each, 100k steps, Adam lr 1e-4 with cosine decay) on one NVIDIA RTX 4060 Ti. Relative L2 error with 3 and 39 context examples:

| Experiment | 3 examples | 39 examples |
|---|---|---|
| Burgers | 0.0453 | 0.0391 |
| 1D Darcy | 0.0866 | 0.0449 |
| 2D Darcy | 0.1289 | 0.1163 |

Reruns on other hardware or seeds will vary slightly. `bash scripts/run_all.sh` regenerates the data and these results.

Not covered by this code:

* The 12-task forward/inverse experiment (Sec. 6.4), the extrapolation experiment (Sec. 6.5, trained with a fixed small prompt) and the prompt ablations (App. C). Their numbers in the paper were not produced by this repository.
* The DeepOSets vs ICON numbers (Sec. 6.6, B.5, D) in the paper come from an earlier run on a different GPU. `experiments/icon_comparison/` reruns them; training times depend on the GPU, and a rerun gives similar but not identical values.

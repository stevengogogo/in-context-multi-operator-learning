# Reviewer Guide

This guide is meant for someone opening the repository for the first time.
The short version is:

```bash
uv sync
bash scripts/run_smoke.sh
```

If the smoke test passes, the package import, data generation, model training,
checkpoint loading, evaluation, plotting, and custom-data path are all working.

## What The Code Solves

The repository contains three paper experiment families.

1D variable-coefficient Poisson:

```text
-(k(x) u_x(x))_x = f(x),  x in (0, 1)
u(0) = u(1) = 0
```

For each operator, `k(x)` is fixed and several right-hand sides `f(x)` are
sampled. The model sees context pairs `(f_i, u_i)` and predicts the solution
for a new query forcing `f_query`.

1D viscous Burgers:

```text
u_t + u u_x = mu u_xx,  x in (0, 1), t in (0, 1]
u(0, t) = u(1, t) = 0
u(x, 0) = u0(x)
```

For each operator, the viscosity `mu` is fixed and several initial conditions
`u0(x)` are sampled. The model predicts the final-time solution from a query
initial condition.

2D variable-coefficient Poisson:

```text
-div(k(x, y) grad u(x, y)) = f(x, y),  (x, y) in (0, 1)^2
u = 0 on the boundary
```

For each operator, `k(x, y)` is fixed and several right-hand sides `f(x, y)`
are sampled. The model predicts the solution for a query forcing field.

## Main Workflow

Use `scripts/run_experiment.py` when you want one entry point:

```bash
python scripts/run_experiment.py --problem poisson1d --stage data
python scripts/run_experiment.py --problem poisson1d --stage train
python scripts/run_experiment.py --problem poisson1d --stage evaluate
python scripts/run_experiment.py --problem poisson1d --stage plot
```

Use the individual scripts when you want explicit control over a stage:

```text
scripts/generate_*.py      Generate train/test .npz datasets
scripts/train_*.py         Train and save .eqx model checkpoints
scripts/evaluate_*.py      Evaluate relative L2 error across context sizes
scripts/plot_*.py          Plot one prediction example
```

Generated data, checkpoints, and figures live under `data/` and `outputs/`.
Both directories are ignored by Git because full experiments are large.

## Code Map

The package code lives in `deeposets/`.

```text
deeposets/configs.py       Paper and custom-model hyperparameter defaults
deeposets/data.py          Dataset metadata, samplers, and custom data import
deeposets/datasets.py      Paper dataset generation functions
deeposets/evaluation.py    Evaluation metrics and context-size sweeps
deeposets/experiments.py   High-level reusable experiment controller
deeposets/models.py        MLP, DeepSet, DeepONet, and DeepOSetsModel modules
deeposets/plotting.py      Context-error and prediction plotting utilities
deeposets/training.py      Jitted prediction, losses, training loop, checkpoints
deeposets/solvers/         Numerical PDE solvers and random field generation
```

## Data Shapes

All model batches are represented by `deeposets.data.ContextData`.

```text
cond_context: (n_eqns, num_ucpair - 1, n_points)
qoi_context:  (n_eqns, num_ucpair - 1, n_points)
cond_query:   (n_eqns, n_points)
xs_query:     (n_points, coordinate_dim)
qoi_answer:   (n_eqns, n_points)
```

Here `cond` means the input condition or forcing, and `qoi` means the quantity
of interest, usually the PDE solution. The sampler randomly selects operators
and condition-output pairs from the saved `.npz` files. Sampling is with
replacement, which matches the original experiment workflow and keeps the
batch sampler simple.

## Custom Data

For custom data, use arrays with shape:

```text
(n_operators, n_pairs, ...)
```

The trailing spatial dimensions are flattened internally. The current
architecture assumes the input field and output field have the same number of
spatial points. Convert custom `.npz` files with:

```bash
python scripts/prepare_custom_dataset.py \
  --source path/to/source_data.npz \
  --input-key f \
  --output-key u \
  --coords-key xs \
  --save-dir data/custom/my_experiment \
  --prefix my_experiment
```

Then use `scripts/train_custom.py`, `scripts/evaluate_custom.py`, and
`scripts/plot_custom_prediction.py`.

## What To Modify

For new PDE data generation, add a solver or generator under `deeposets/solvers`
or `deeposets/datasets.py`, then add a script in `scripts/`.

For a new model architecture, start in `deeposets/models.py` and keep the
training API in `deeposets/training.py` unchanged if possible.

For new evaluation plots, add reusable plotting code in `deeposets/plotting.py`
and keep command-line scripts as thin wrappers.

For most user-provided datasets, no package edits are needed. Use the custom
data scripts first.

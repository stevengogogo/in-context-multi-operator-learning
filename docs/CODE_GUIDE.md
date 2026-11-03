# DeepOSets: Code Guide

This repository contains the JAX implementation used to reproduce the DeepOSets experiments for 1D Poisson, 1D viscous Burgers, and 2D variable-coefficient Poisson problems.

The code is organized as a small Python package plus command-line scripts. The numerical solvers, data samplers, neural operator model, training loop, and evaluation routines live in `deeposets/`. The scripts in `scripts/` run the paper workflows with the same default seeds and experiment sizes as the original experiments.

If you are reviewing the code for the first time, read `docs/REVIEWER_GUIDE.md` after the smoke test. It gives the equation summary, data shapes, code map, and extension points in one place.

## Repository Layout

```text
deeposets/
  data.py                 ContextData and dataset samplers
  experiments.py          Reusable experiment controller for data/train/evaluate/plot
  models.py               MLP, DeepSet, DeepONet, DeepOSetsModel
  training.py             Losses, prediction, checkpointing, training loop
  plotting.py             Reusable plotting helpers
  datasets.py             Dataset generation workflows
  configs.py              Default experiment hyperparameters
  solvers/
    one_d.py              1D Poisson, Burgers, ODE, and GP utilities
    two_d.py              2D Poisson FEM solver and random fields
scripts/
  generate_1d_poisson.py
  generate_1d_burgers.py
  generate_2d_poisson.py
  train_1d.py
  train_2d.py
  evaluate_1d.py
  evaluate_2d.py
  plot_1d_prediction.py
  plot_2d_prediction.py
  prepare_custom_dataset.py
  train_custom.py
  evaluate_custom.py
  plot_custom_prediction.py
  run_experiment.py       One modular entry point for all workflow stages
docs/
  REVIEWER_GUIDE.md       Equations, data shapes, code map, and extension notes
  CODE_GUIDE.md           This file
tests/
  smoke_test.py           Fast end-to-end sanity checks
experiments/icon_comparison/   PyTorch DeepOSets vs ICON benchmark (see its README)
third_party/icon-tutorial/     Vendored ICON reference code (own LICENSE)
```

## Environment

Dependencies are managed with [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync                  # creates .venv from uv.lock
uv sync --extra icon     # also installs PyTorch for the ICON comparison
```

Run any script with `uv run`, for example `uv run python scripts/train_1d.py --help`, or activate `.venv` yourself.

## Smoke Test

Before starting the full experiments, run the smoke test. It creates tiny temporary datasets, trains each path for one step, and evaluates the saved checkpoints.

```bash
bash scripts/run_smoke.sh
```

If this finishes with `Smoke test finished successfully.`, the repository, package imports, JAX installation, model code, dataset generation, training, checkpointing, and evaluation are all wired correctly.

For a code-oriented walkthrough after the smoke test, see `docs/REVIEWER_GUIDE.md`.

## Mathematical Problems

The experiments cover three operator-learning settings.

1D variable-coefficient Poisson:

```text
-(k(x) u_x(x))_x = f(x),  u(0) = u(1) = 0
```

1D viscous Burgers:

```text
u_t + u u_x = mu u_xx,  u(0, t) = u(1, t) = 0
```

2D variable-coefficient Poisson:

```text
-div(k(x, y) grad u(x, y)) = f(x, y),  u = 0 on the boundary
```

In each experiment, the model receives several context input-output pairs for an operator and predicts the output for a new query input from the same operator.

## Modular Workflow

The code can be run stage-by-stage. This is useful when data or trained checkpoints already exist and you do not want to regenerate data or retrain the model.

```bash
python scripts/run_experiment.py --problem poisson1d --stage data
python scripts/run_experiment.py --problem poisson1d --stage train
python scripts/run_experiment.py --problem poisson1d --stage evaluate
python scripts/run_experiment.py --problem poisson1d --stage plot
```

Supported problems are:

```bash
poisson1d
burgers1d
poisson2d
```

The combined command below runs the whole workflow, but reuses existing datasets and trained models by default:

```bash
python scripts/run_experiment.py --problem poisson1d --stage all
```

To force regeneration and retraining, add `--force`:

```bash
python scripts/run_experiment.py --problem poisson1d --stage all --force
```

For quick local checks, override the data and training sizes:

```bash
python scripts/run_experiment.py --problem poisson1d --stage all --n-operators 4 --chunk-size 2 --n-pairs 4 --N 8 --steps 2 --min-ucpair 3 --max-ucpair 3 --n-trials 1 --eval-n-eqns 2 --plot-n-eqns 2 --num-ucpair 3
```

The individual training scripts also support reusing existing checkpoints:

```bash
python scripts/train_1d.py --problem poisson --reuse-existing
python scripts/train_2d.py --reuse-existing
```

## Python API

Readers can import data, samplers, trained models, evaluation, and plotting without running the full training workflow again.

```python
import jax.random as jr
from deeposets import DeepOSetExperiment, describe_dataset, load_dataset

experiment = DeepOSetExperiment("poisson1d")
print(experiment.paths())

arrays = load_dataset(experiment.test_dataset_path)
info = describe_dataset(experiment.test_dataset_path)
print(info)

sampler = experiment.load_sampler(split="test", n_eqns=4, num_ucpair=5)
batch = sampler(jr.PRNGKey(0))

model = experiment.load_model()
results = experiment.evaluate(n_trials=10, min_ucpair=3, max_ucpair=10)
experiment.plot_prediction(num_ucpair=5, n_eqns=4)
```

This API is the cleanest way to reuse generated datasets and trained checkpoints in a notebook or a new script. Data generation, training, evaluation, and plotting are independent methods on `DeepOSetExperiment`.

## Custom Data

You can train and evaluate on your own operator data without editing the paper experiment scripts. The custom workflow expects input/query fields and output/solution fields with shape:

```text
(n_operators, n_pairs, ...)
```

Within each sampled batch, `num_ucpair - 1` pairs are used as in-context examples and one sampled pair is used as the query. The trailing spatial dimensions are flattened internally. The current model assumes the input field and output field use the same number of spatial points.

The sampler draws operators and pairs with replacement, matching the paper workflow. Coordinates can be provided as `(n_points,)`, `(n_points, coordinate_dim)`, or as a grid with shape `spatial_shape + (coordinate_dim,)`. If `xs` is omitted, a unit-interval or unit-square coordinate grid is created automatically from the field shape.

If your data is already in Python arrays, save it directly:

```python
from deeposets import save_custom_operator_dataset

train_path, test_path = save_custom_operator_dataset(
    save_dir="data/custom/my_experiment",
    cond=my_input_array,
    qoi=my_output_array,
    xs=my_coordinates,
    train_fraction=0.9,
    shuffle=True,
    seed=0,
)
```

If your data is in an `.npz` file, convert it once. In this example the source file has arrays named `f`, `u`, and `xs`:

```bash
python scripts/prepare_custom_dataset.py \
  --source path/to/my_data.npz \
  --input-key f \
  --output-key u \
  --coords-key xs \
  --save-dir data/custom/my_experiment \
  --prefix my_experiment \
  --shuffle
```

This creates:

```text
data/custom/my_experiment/my_experiment_dataset_train.npz
data/custom/my_experiment/my_experiment_dataset_test.npz
```

Train on the custom dataset:

```bash
python scripts/train_custom.py \
  --train-dataset data/custom/my_experiment/my_experiment_dataset_train.npz \
  --save-dir outputs/custom/my_experiment \
  --steps 10000 \
  --n-eqns 5 \
  --num-ucpair 5
```

Evaluate the saved custom model without retraining:

```bash
python scripts/evaluate_custom.py \
  --test-dataset data/custom/my_experiment/my_experiment_dataset_test.npz \
  --model outputs/custom/my_experiment/deeposets_custom.eqx \
  --min-ucpair 3 \
  --max-ucpair 5 \
  --n-trials 100
```

Plot one custom prediction:

```bash
python scripts/plot_custom_prediction.py \
  --test-dataset data/custom/my_experiment/my_experiment_dataset_test.npz \
  --model outputs/custom/my_experiment/deeposets_custom.eqx \
  --num-ucpair 5
```

The custom training script saves `custom_model_config.json` next to the model. The evaluation and plotting scripts load that file automatically, so you do not need to repeat the architecture options as long as the config stays next to `deeposets_custom.eqx`.

## Full Experiments

The defaults below match the experiment code: seed `0`, 200,000 operators (50,000 for 2D Darcy), and the same grid sizes and context-pair counts used in the original runs. These datasets are large, so the generated files are intentionally ignored by Git.

### 1D Poisson

Generate data:

```bash
python scripts/generate_1d_poisson.py
```

Train:

```bash
python scripts/train_1d.py --problem poisson
```

Evaluate:

```bash
python scripts/evaluate_1d.py --problem poisson
```

Plot one prediction:

```bash
python scripts/plot_1d_prediction.py --problem poisson
```

Outputs:

```bash
data/one_dimension/poisson/
outputs/one_dimension/poisson/
outputs/evaluation/poisson_context_error.png
outputs/figures/poisson1d_prediction.png
```

### 1D Burgers

Generate data:

```bash
python scripts/generate_1d_burgers.py
```

Train:

```bash
python scripts/train_1d.py --problem burgers
```

Evaluate:

```bash
python scripts/evaluate_1d.py --problem burgers
```

Plot one prediction:

```bash
python scripts/plot_1d_prediction.py --problem burgers
```

Outputs:

```bash
data/one_dimension/burgers/
outputs/one_dimension/burgers/
outputs/evaluation/burgers_context_error.png
outputs/figures/burgers1d_prediction.png
```

### 2D Poisson

Generate data:

```bash
python scripts/generate_2d_poisson.py
```

Train:

```bash
python scripts/train_2d.py
```

Evaluate:

```bash
python scripts/evaluate_2d.py
```

Plot one prediction:

```bash
python scripts/plot_2d_prediction.py
```

Outputs:

```bash
data/two_dimension/poisson/
outputs/two_dimension/poisson/
outputs/evaluation/poisson2d_context_error.png
outputs/figures/poisson2d_prediction.png
```

## Plotting

There are two kinds of plots.

Context-size plots show how accuracy changes as the number of in-context examples changes. These are created by the evaluation scripts:

```bash
python scripts/evaluate_1d.py --problem poisson
python scripts/evaluate_1d.py --problem burgers
python scripts/evaluate_2d.py
```

Prediction plots show a single sampled operator and compare the true and predicted query solution. These are created by:

```bash
python scripts/plot_1d_prediction.py --problem poisson
python scripts/plot_1d_prediction.py --problem burgers
python scripts/plot_2d_prediction.py
```

To choose a different random example, change `--seed`. To choose a different operator inside the sampled batch, change `--operator-index`. To control the number of context pairs shown to the model, change `--num-ucpair`.

Examples:

```bash
python scripts/plot_1d_prediction.py --problem poisson --seed 7 --operator-index 0 --num-ucpair 10
python scripts/plot_2d_prediction.py --seed 7 --operator-index 0 --num-ucpair 10
```

## Quick Debug Runs

Use these commands when testing changes locally. They run the same code paths with tiny datasets and very short training.

```bash
python scripts/generate_1d_poisson.py --save-dir data/debug/poisson --n-operators 4 --chunk-size 2 --n-pairs 4 --N 8
python scripts/train_1d.py --problem poisson --dataset data/debug/poisson/poisson1d_dataset_train.npz --save-dir outputs/debug/poisson --steps 2 --save-every 1
python scripts/evaluate_1d.py --problem poisson --dataset data/debug/poisson/poisson1d_dataset_test.npz --model outputs/debug/poisson/poisson/deeposets_model_poisson.eqx --min-ucpair 3 --max-ucpair 3 --n-trials 1 --n-eqns 2
python scripts/plot_1d_prediction.py --problem poisson --dataset data/debug/poisson/poisson1d_dataset_test.npz --model outputs/debug/poisson/poisson/deeposets_model_poisson.eqx --num-ucpair 3 --n-eqns 2 --save-plot outputs/debug/poisson_prediction.png
```

```bash
python scripts/generate_2d_poisson.py --save-dir data/debug/poisson2d --n-operators 2 --chunk-size 1 --n-pairs 2 --N 4
python scripts/train_2d.py --dataset data/debug/poisson2d/poisson2d_dataset_train.npz --save-dir outputs/debug/poisson2d --steps 2 --save-every 1
python scripts/evaluate_2d.py --dataset data/debug/poisson2d/poisson2d_dataset_test.npz --model outputs/debug/poisson2d/deeposets_model_poisson2d.eqx --min-ucpair 3 --max-ucpair 3 --n-trials 1 --n-eqns 2
python scripts/plot_2d_prediction.py --dataset data/debug/poisson2d/poisson2d_dataset_test.npz --model outputs/debug/poisson2d/deeposets_model_poisson2d.eqx --num-ucpair 3 --n-eqns 2 --save-plot outputs/debug/poisson2d_prediction.png
```

## Notes On Reproducibility

The command-line scripts preserve the original random seeds and default model sizes. The data loaders keep full datasets as NumPy arrays on CPU and transfer only sampled batches to JAX, which avoids holding the full dataset in accelerator memory. The training code stores Python floats for losses rather than JAX arrays, preventing long-run memory growth.

The supported workflows are the package modules and scripts documented above.

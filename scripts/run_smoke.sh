#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/deeposets_smoke.XXXXXX")"
echo "Writing smoke-test artifacts to: $TMP_DIR"

uv run python "$ROOT_DIR/tests/smoke_test.py"

uv run python "$ROOT_DIR/scripts/generate_1d_poisson.py" \
  --save-dir "$TMP_DIR/poisson1d" \
  --n-operators 4 \
  --chunk-size 2 \
  --n-pairs 4 \
  --N 8

uv run python "$ROOT_DIR/scripts/train_1d.py" \
  --problem poisson \
  --dataset "$TMP_DIR/poisson1d/poisson1d_dataset_train.npz" \
  --save-dir "$TMP_DIR/outputs_1d" \
  --steps 1 \
  --save-every 1

uv run python "$ROOT_DIR/scripts/evaluate_1d.py" \
  --problem poisson \
  --dataset "$TMP_DIR/poisson1d/poisson1d_dataset_test.npz" \
  --model "$TMP_DIR/outputs_1d/poisson/deeposets_model_poisson.eqx" \
  --min-ucpair 3 \
  --max-ucpair 3 \
  --n-trials 1 \
  --n-eqns 2 \
  --save-plot "$TMP_DIR/poisson1d_eval.png" \
  --save-metrics "$TMP_DIR/poisson1d_eval_metrics.npz"

uv run python "$ROOT_DIR/scripts/plot_1d_prediction.py" \
  --problem poisson \
  --dataset "$TMP_DIR/poisson1d/poisson1d_dataset_test.npz" \
  --model "$TMP_DIR/outputs_1d/poisson/deeposets_model_poisson.eqx" \
  --num-ucpair 3 \
  --n-eqns 2 \
  --save-plot "$TMP_DIR/poisson1d_prediction.png"

uv run python "$ROOT_DIR/scripts/generate_2d_poisson.py" \
  --save-dir "$TMP_DIR/poisson2d" \
  --n-operators 2 \
  --chunk-size 1 \
  --n-pairs 2 \
  --N 4

uv run python "$ROOT_DIR/scripts/train_2d.py" \
  --dataset "$TMP_DIR/poisson2d/poisson2d_dataset_train.npz" \
  --save-dir "$TMP_DIR/outputs_2d" \
  --steps 1 \
  --save-every 1

uv run python "$ROOT_DIR/scripts/evaluate_2d.py" \
  --dataset "$TMP_DIR/poisson2d/poisson2d_dataset_test.npz" \
  --model "$TMP_DIR/outputs_2d/deeposets_model_poisson2d.eqx" \
  --min-ucpair 3 \
  --max-ucpair 3 \
  --n-trials 1 \
  --n-eqns 2 \
  --save-plot "$TMP_DIR/poisson2d_eval.png" \
  --save-metrics "$TMP_DIR/poisson2d_eval_metrics.npz"

uv run python "$ROOT_DIR/scripts/plot_2d_prediction.py" \
  --dataset "$TMP_DIR/poisson2d/poisson2d_dataset_test.npz" \
  --model "$TMP_DIR/outputs_2d/deeposets_model_poisson2d.eqx" \
  --num-ucpair 3 \
  --n-eqns 2 \
  --save-plot "$TMP_DIR/poisson2d_prediction.png"

uv run python - "$TMP_DIR/custom_source.npz" <<'PY'
import sys
import numpy as np

rng = np.random.default_rng(0)
xs = np.linspace(0.0, 1.0, 8, dtype=np.float32)
inputs = rng.normal(size=(4, 4, 8)).astype(np.float32)
outputs = np.cumsum(inputs, axis=-1).astype(np.float32) / 8.0
np.savez(sys.argv[1], input_field=inputs, solution=outputs, xs=xs)
PY

uv run python "$ROOT_DIR/scripts/prepare_custom_dataset.py" \
  --source "$TMP_DIR/custom_source.npz" \
  --input-key input_field \
  --output-key solution \
  --coords-key xs \
  --save-dir "$TMP_DIR/custom" \
  --prefix custom

uv run python "$ROOT_DIR/scripts/train_custom.py" \
  --train-dataset "$TMP_DIR/custom/custom_dataset_train.npz" \
  --save-dir "$TMP_DIR/outputs_custom" \
  --steps 1 \
  --save-every 1 \
  --n-eqns 2 \
  --num-ucpair 3 \
  --width 16 \
  --depth 1

uv run python "$ROOT_DIR/scripts/evaluate_custom.py" \
  --test-dataset "$TMP_DIR/custom/custom_dataset_test.npz" \
  --model "$TMP_DIR/outputs_custom/deeposets_custom.eqx" \
  --min-ucpair 3 \
  --max-ucpair 3 \
  --n-trials 1 \
  --n-eqns 2 \
  --save-plot "$TMP_DIR/custom_eval.png" \
  --save-metrics "$TMP_DIR/custom_eval_metrics.npz"

uv run python "$ROOT_DIR/scripts/plot_custom_prediction.py" \
  --test-dataset "$TMP_DIR/custom/custom_dataset_test.npz" \
  --model "$TMP_DIR/outputs_custom/deeposets_custom.eqx" \
  --num-ucpair 3 \
  --n-eqns 2 \
  --save-plot "$TMP_DIR/custom_prediction.png"

echo "Smoke test finished successfully."

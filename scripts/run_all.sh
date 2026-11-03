#!/usr/bin/env bash
# Re-run all experiments. Data goes to data/, results to experiments/output/.
# Usage: bash scripts/run_all.sh   (or STAGES="burgers darcy1d" bash scripts/run_all.sh)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=experiments/output
DATA=data
STAGES="${STAGES:-darcy2d burgers darcy1d icon}"
STEPS="${STEPS:-100000}"
LR="${LR:-1e-4}"
mkdir -p "$OUT/logs"

run() { uv run python "$@"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

pipeline_1d() {
  local problem=$1 dir=$OUT/$2 data=$DATA/one_dimension/$1
  if [[ $problem == burgers ]]; then
    run scripts/generate_1d_burgers.py --save-dir "$data" --n-pairs 40 --reuse-existing
  else
    run scripts/generate_1d_poisson.py --save-dir "$data" --n-pairs 40 --reuse-existing
  fi
  run scripts/train_1d.py --problem "$problem" --dataset "$data/${problem}1d_dataset_train.npz" \
    --save-dir "$dir" --steps "$STEPS" --lr "$LR" --reuse-existing
  local model=$dir/$problem/deeposets_model_$problem.eqx
  run scripts/evaluate_1d.py --problem "$problem" --dataset "$data/${problem}1d_dataset_test.npz" \
    --model "$model" --min-ucpair 4 --max-ucpair 40 --n-eqns 100 \
    --save-plot "$dir/context_error.png" --save-metrics "$dir/metrics.npz"
  run scripts/plot_1d_prediction.py --problem "$problem" --dataset "$data/${problem}1d_dataset_test.npz" \
    --model "$model" --save-plot "$dir/prediction.png"
}

pipeline_2d() {
  local dir=$OUT/darcy2d data=$DATA/two_dimension/poisson
  run scripts/generate_2d_poisson.py --save-dir "$data" --n-operators 50000 --n-pairs 40 --reuse-existing
  run scripts/train_2d.py --dataset "$data/poisson2d_dataset_train.npz" --save-dir "$dir" \
    --steps "$STEPS" --lr "$LR" --reuse-existing
  run scripts/evaluate_2d.py --dataset "$data/poisson2d_dataset_test.npz" \
    --model "$dir/deeposets_model_poisson2d.eqx" --min-ucpair 4 --max-ucpair 40 --n-eqns 100 \
    --save-plot "$dir/context_error.png" --save-metrics "$dir/metrics.npz"
  run scripts/plot_2d_prediction.py --dataset "$data/poisson2d_dataset_test.npz" \
    --model "$dir/deeposets_model_poisson2d.eqx" --save-plot "$dir/prediction.png"
}

pipeline_icon() {
  local dir=$OUT/icon_comparison cmp=experiments/icon_comparison
  mkdir -p "$dir"
  [[ -f $DATA/icon_comparison.npz ]] || run $cmp/datagen.py
  for steps in 10000 100000; do
    suffix=""; [[ $steps == 10000 ]] && suffix=_10k
    N_STEPS=$steps OUT_FILE=$dir/deeposets_results$suffix.json run $cmp/train_deeposets.py
    N_STEPS=$steps OUT_FILE=$dir/icon_results$suffix.json run $cmp/train_icon.py
  done
  run $cmp/nn_baseline.py
  run $cmp/dataset_ablation.py
}

if has darcy2d; then
  pipeline_2d 2>&1 | tee "$OUT/logs/darcy2d.log" &
  PID_2D=$!
fi
if has burgers; then pipeline_1d burgers burgers 2>&1 | tee "$OUT/logs/burgers.log" & fi
[[ -n ${PID_2D:-} ]] && wait "$PID_2D"
if has darcy1d; then pipeline_1d poisson darcy1d 2>&1 | tee "$OUT/logs/darcy1d.log"; fi
wait
if has icon; then pipeline_icon 2>&1 | tee "$OUT/logs/icon_comparison.log"; fi

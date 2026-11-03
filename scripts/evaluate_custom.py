"""Evaluate a trained custom DeepOSets model across context sizes."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import equinox as eqx
import jax
import jax.random as jr
import numpy as np

from deeposets.configs import custom_experiment_config
from deeposets.data import get_data_sampler
from deeposets.evaluation import evaluate_context_sweep
from deeposets.models import DeepOSetsModel
from deeposets.plotting import plot_context_error

jax.config.update("jax_enable_x64", False)


def load_training_config(config_path):
    """Load architecture options saved by `train_custom.py`, if available."""
    if config_path and config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def main():
    """Evaluate a custom checkpoint and save context-size metrics and a plot."""
    parser = argparse.ArgumentParser(description="Evaluate a trained custom DeepOSets model.")
    parser.add_argument("--test-dataset", type=Path, default=Path("data/custom/custom_dataset_test.npz"))
    parser.add_argument("--model", type=Path, default=Path("outputs/custom/deeposets_custom.eqx"))
    parser.add_argument("--config", type=Path, default=None)

    parser.add_argument("--min-ucpair", type=int, default=3)
    parser.add_argument("--max-ucpair", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-eqns", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-plot", type=Path, default=Path("outputs/evaluation/custom_context_error.png"))
    parser.add_argument("--save-metrics", type=Path, default=Path("outputs/evaluation/custom_context_error_metrics.npz"))

    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-embed-h", type=int, default=None)
    parser.add_argument("--n-intermediate", type=int, default=None)
    parser.add_argument("--pooling", choices=["mean", "sum", "max"], default="mean")
    args = parser.parse_args()

    config_path = args.config or args.model.with_name("custom_model_config.json")
    run_config = load_training_config(config_path)
    width = run_config.get("width", args.width)
    depth = run_config.get("depth", args.depth)
    n_embed_h = run_config.get("n_embed_h", args.n_embed_h)
    n_intermediate = run_config.get("n_intermediate", args.n_intermediate)
    pooling = run_config.get("pooling", args.pooling)
    n_eqns = args.n_eqns or run_config.get("n_eqns", 5)
    max_ucpair = args.max_ucpair or run_config.get("num_ucpair", 5)

    config = custom_experiment_config(
        width=width,
        depth=depth,
        n_embed_h=n_embed_h,
        n_intermediate=n_intermediate,
        pooling=pooling,
    )
    template_sampler = get_data_sampler(args.test_dataset, n_eqns, max(args.min_ucpair, 3))
    template_batch = template_sampler(jr.PRNGKey(0))
    model_template = DeepOSetsModel.from_data(template_batch, **config["model_param"], key=jr.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(args.model, model_template)

    context_sizes = range(args.min_ucpair, max_ucpair + 1)

    def sampler_factory(num_ucpair):
        return get_data_sampler(args.test_dataset, n_eqns, num_ucpair)

    results = evaluate_context_sweep(model, sampler_factory, context_sizes, n_trials=args.n_trials, seed=args.seed)
    for num_ucpair, rel_l2, mse in zip(results["context_sizes"], results["relative_l2"], results["mse"]):
        print(f"ucpair={int(num_ucpair):2d}: rel_L2={rel_l2:.4f}, MSE={mse:.2e}")

    plot_context_error(results["context_sizes"], results["relative_l2"], "Accuracy vs Context Size - Custom", args.save_plot)
    args.save_metrics.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.save_metrics, **results)
    print(f"Saved plot to {args.save_plot}")
    print(f"Saved metrics to {args.save_metrics}")


if __name__ == "__main__":
    main()

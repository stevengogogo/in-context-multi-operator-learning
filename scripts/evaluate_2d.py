"""Evaluate a trained 2D Poisson DeepOSets model across context sizes."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import equinox as eqx
import jax
import jax.random as jr
import numpy as np

from deeposets.configs import two_dim_experiment_config
from deeposets.data import get_2d_data_sampler
from deeposets.evaluation import evaluate_context_sweep
from deeposets.models import DeepOSetsModel
from deeposets.plotting import plot_context_error

jax.config.update("jax_enable_x64", False)


def main():
    """Evaluate a 2D checkpoint and save context-size metrics and a plot."""
    parser = argparse.ArgumentParser(description="Evaluate a trained 2D Poisson DeepOSets model.")
    parser.add_argument("--dataset", type=Path, default=Path("data/two_dimension/poisson/poisson2d_dataset_test.npz"))
    parser.add_argument("--model", type=Path, default=Path("outputs/two_dimension/poisson/deeposets_model_poisson2d.eqx"))
    parser.add_argument("--min-ucpair", type=int, default=3)
    parser.add_argument("--max-ucpair", type=int, default=29)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-eqns", type=int, default=20)
    parser.add_argument("--save-plot", type=Path, default=None)
    parser.add_argument("--save-metrics", type=Path, default=None)
    args = parser.parse_args()

    config = two_dim_experiment_config()
    template_sampler = get_2d_data_sampler(args.dataset, args.n_eqns, max(args.min_ucpair, 3))
    template_batch = template_sampler(jr.PRNGKey(0))
    model_template = DeepOSetsModel.from_data(template_batch, **config["model_param"], key=jr.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(args.model, model_template)

    context_sizes = range(args.min_ucpair, args.max_ucpair + 1)

    def sampler_factory(num_ucpair):
        return get_2d_data_sampler(args.dataset, args.n_eqns, num_ucpair)

    results = evaluate_context_sweep(model, sampler_factory, context_sizes, n_trials=args.n_trials, seed=42)
    for num_ucpair, rel_l2, mse in zip(results["context_sizes"], results["relative_l2"], results["mse"]):
        print(f"ucpair={int(num_ucpair):2d}: rel_L2={rel_l2:.4f}, MSE={mse:.2e}")

    save_plot = args.save_plot or Path("outputs/evaluation/poisson2d_context_error.png")
    plot_context_error(
        results["context_sizes"],
        results["relative_l2"],
        "Accuracy vs Context Size - 2D Poisson",
        save_plot,
    )
    save_metrics = args.save_metrics or Path("outputs/evaluation/poisson2d_context_error_metrics.npz")
    save_metrics.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_metrics, **results)
    print(f"Saved plot to {save_plot}")
    print(f"Saved metrics to {save_metrics}")


if __name__ == "__main__":
    main()

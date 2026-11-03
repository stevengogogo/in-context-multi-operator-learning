"""Evaluate a trained 1D DeepOSets model across context sizes."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import equinox as eqx
import jax
import jax.random as jr
import numpy as np

from deeposets.configs import one_dim_experiment_config
from deeposets.data import get_1d_data_sampler
from deeposets.evaluation import evaluate_context_sweep
from deeposets.models import DeepOSetsModel
from deeposets.plotting import plot_context_error

jax.config.update("jax_enable_x64", False)


def default_paths(problem):
    """Return default test dataset and checkpoint paths for a 1D problem."""
    if problem == "poisson":
        return (
            Path("data/one_dimension/poisson/poisson1d_dataset_test.npz"),
            Path("outputs/one_dimension/poisson/deeposets_model_poisson.eqx"),
        )
    return (
        Path("data/one_dimension/burgers/burgers1d_dataset_test.npz"),
        Path("outputs/one_dimension/burgers/deeposets_model_burgers.eqx"),
    )


def main():
    """Evaluate a 1D checkpoint and save context-size metrics and a plot."""
    parser = argparse.ArgumentParser(description="Evaluate a trained 1D DeepOSets model.")
    parser.add_argument("--problem", choices=["poisson", "burgers"], default="poisson")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--min-ucpair", type=int, default=3)
    parser.add_argument("--max-ucpair", type=int, default=39)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-eqns", type=int, default=20)
    parser.add_argument("--save-plot", type=Path, default=None)
    parser.add_argument("--save-metrics", type=Path, default=None)
    args = parser.parse_args()

    dataset_default, model_default = default_paths(args.problem)
    dataset_path = args.dataset or dataset_default
    model_path = args.model or model_default
    config = one_dim_experiment_config()

    template_sampler = get_1d_data_sampler(dataset_path, args.n_eqns, max(args.min_ucpair, 3))
    template_batch = template_sampler(jr.PRNGKey(0))
    model_template = DeepOSetsModel.from_data(template_batch, **config["model_param"], key=jr.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(model_path, model_template)

    context_sizes = range(args.min_ucpair, args.max_ucpair + 1)

    def sampler_factory(num_ucpair):
        return get_1d_data_sampler(dataset_path, args.n_eqns, num_ucpair)

    results = evaluate_context_sweep(model, sampler_factory, context_sizes, n_trials=args.n_trials, seed=42)
    for num_ucpair, rel_l2, mse in zip(results["context_sizes"], results["relative_l2"], results["mse"]):
        print(f"ucpair={int(num_ucpair):2d}: rel_L2={rel_l2:.4f}, MSE={mse:.2e}")

    save_plot = args.save_plot or Path(f"outputs/evaluation/{args.problem}_context_error.png")
    plot_context_error(
        results["context_sizes"],
        results["relative_l2"],
        f"Accuracy vs Context Size - 1D {args.problem.title()}",
        save_plot,
    )
    save_metrics = args.save_metrics or Path(f"outputs/evaluation/{args.problem}_context_error_metrics.npz")
    save_metrics.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_metrics, **results)
    print(f"Saved plot to {save_plot}")
    print(f"Saved metrics to {save_metrics}")


if __name__ == "__main__":
    main()

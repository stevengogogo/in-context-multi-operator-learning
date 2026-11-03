"""Plot one prediction from a trained 1D DeepOSets model."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import equinox as eqx
import jax
import jax.random as jr

from deeposets.configs import one_dim_experiment_config
from deeposets.data import get_1d_data_sampler
from deeposets.models import DeepOSetsModel
from deeposets.plotting import plot_1d_prediction

jax.config.update("jax_enable_x64", False)


def default_paths(problem):
    """Return default test dataset, checkpoint, and plot paths."""
    if problem == "poisson":
        return (
            Path("data/one_dimension/poisson/poisson1d_dataset_test.npz"),
            Path("outputs/one_dimension/poisson/deeposets_model_poisson.eqx"),
            Path("outputs/figures/poisson1d_prediction.png"),
        )
    return (
        Path("data/one_dimension/burgers/burgers1d_dataset_test.npz"),
        Path("outputs/one_dimension/burgers/deeposets_model_burgers.eqx"),
        Path("outputs/figures/burgers1d_prediction.png"),
    )


def main():
    """Load a 1D checkpoint and save one prediction comparison plot."""
    parser = argparse.ArgumentParser(description="Plot one 1D prediction from a trained model.")
    parser.add_argument("--problem", choices=["poisson", "burgers"], default="poisson")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--num-ucpair", type=int, default=5)
    parser.add_argument("--n-eqns", type=int, default=4)
    parser.add_argument("--operator-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--save-plot", type=Path, default=None)
    args = parser.parse_args()

    default_dataset, default_model, default_plot = default_paths(args.problem)
    dataset_path = args.dataset or default_dataset
    model_path = args.model or default_model
    save_plot = args.save_plot or default_plot

    config = one_dim_experiment_config()
    sampler = get_1d_data_sampler(dataset_path, args.n_eqns, args.num_ucpair)
    data = sampler(jr.PRNGKey(args.seed))
    template = DeepOSetsModel.from_data(data, **config["model_param"], key=jr.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(model_path, template)
    plot_1d_prediction(model, data, args.problem, save_plot, operator_index=args.operator_index)
    print(f"Saved plot to {save_plot}")


if __name__ == "__main__":
    main()

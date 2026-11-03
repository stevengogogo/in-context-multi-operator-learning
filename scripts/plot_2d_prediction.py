"""Plot one prediction from a trained 2D Poisson DeepOSets model."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import equinox as eqx
import jax
import jax.random as jr

from deeposets.configs import two_dim_experiment_config
from deeposets.data import get_2d_data_sampler
from deeposets.models import DeepOSetsModel
from deeposets.plotting import plot_2d_prediction

jax.config.update("jax_enable_x64", False)


def main():
    """Load a 2D checkpoint and save one prediction comparison plot."""
    parser = argparse.ArgumentParser(description="Plot one 2D Poisson prediction from a trained model.")
    parser.add_argument("--dataset", type=Path, default=Path("data/two_dimension/poisson/poisson2d_dataset_test.npz"))
    parser.add_argument("--model", type=Path, default=Path("outputs/two_dimension/poisson/deeposets_model_poisson2d.eqx"))
    parser.add_argument("--num-ucpair", type=int, default=5)
    parser.add_argument("--n-eqns", type=int, default=4)
    parser.add_argument("--operator-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--save-plot", type=Path, default=Path("outputs/figures/poisson2d_prediction.png"))
    args = parser.parse_args()

    config = two_dim_experiment_config()
    sampler = get_2d_data_sampler(args.dataset, args.n_eqns, args.num_ucpair)
    data = sampler(jr.PRNGKey(args.seed))
    template = DeepOSetsModel.from_data(data, **config["model_param"], key=jr.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(args.model, template)
    plot_2d_prediction(model, data, args.save_plot, operator_index=args.operator_index)
    print(f"Saved plot to {args.save_plot}")


if __name__ == "__main__":
    main()

"""Plot one prediction from a trained custom DeepOSets model."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import equinox as eqx
import jax
import jax.random as jr

from deeposets.configs import custom_experiment_config
from deeposets.data import get_data_sampler
from deeposets.models import DeepOSetsModel
from deeposets.plotting import plot_custom_prediction

jax.config.update("jax_enable_x64", False)


def load_training_config(config_path):
    """Load architecture options saved by `train_custom.py`, if available."""
    if config_path and config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def main():
    """Load a custom checkpoint and save one prediction comparison plot."""
    parser = argparse.ArgumentParser(description="Plot one custom operator prediction from a trained model.")
    parser.add_argument("--test-dataset", type=Path, default=Path("data/custom/custom_dataset_test.npz"))
    parser.add_argument("--model", type=Path, default=Path("outputs/custom/deeposets_custom.eqx"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--num-ucpair", type=int, default=None)
    parser.add_argument("--n-eqns", type=int, default=None)
    parser.add_argument("--operator-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--save-plot", type=Path, default=Path("outputs/figures/custom_prediction.png"))

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
    num_ucpair = args.num_ucpair or run_config.get("num_ucpair", 5)

    config = custom_experiment_config(
        width=width,
        depth=depth,
        n_embed_h=n_embed_h,
        n_intermediate=n_intermediate,
        pooling=pooling,
    )
    sampler = get_data_sampler(args.test_dataset, n_eqns, num_ucpair)
    data = sampler(jr.PRNGKey(args.seed))
    template = DeepOSetsModel.from_data(data, **config["model_param"], key=jr.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(args.model, template)
    plot_custom_prediction(model, data, args.save_plot, operator_index=args.operator_index)
    print(f"Saved plot to {args.save_plot}")


if __name__ == "__main__":
    main()

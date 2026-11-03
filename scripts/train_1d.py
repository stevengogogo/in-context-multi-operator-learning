"""Train the DeepOSets model on a generated 1D dataset."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.random as jr
import numpy as np
import optax

from deeposets.configs import one_dim_experiment_config
from deeposets.data import get_1d_data_sampler
from deeposets.models import DeepOSetsModel, n_param
from deeposets.training import save_model, train_model

jax.config.update("jax_enable_x64", False)


def default_dataset_path(problem):
    """Return the default generated training dataset for a 1D problem."""
    if problem == "poisson":
        return Path("data/one_dimension/poisson/poisson1d_dataset_train.npz")
    if problem == "burgers":
        return Path("data/one_dimension/burgers/burgers1d_dataset_train.npz")
    raise ValueError(f"Unsupported problem: {problem}")


def main():
    """Parse CLI arguments, build the model, train, and save outputs."""
    parser = argparse.ArgumentParser(description="Train DeepOSets on a 1D dataset.")
    parser.add_argument("--problem", choices=["poisson", "burgers"], default="poisson")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/one_dimension"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate (default: from config).")
    parser.add_argument("--schedule", choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--reuse-existing", action="store_true", help="Skip training when the final model already exists.")
    args = parser.parse_args()

    config = one_dim_experiment_config()
    if args.steps is not None:
        config["training_param"]["n_step"] = args.steps
    if args.lr is not None:
        config["training_param"]["init_lr"] = args.lr

    dataset_path = args.dataset or default_dataset_path(args.problem)
    save_dir = args.save_dir / args.problem
    save_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = save_dir / f"deeposets_model_{args.problem}.eqx"
    if args.reuse_existing and final_model_path.exists():
        print(f"Using existing trained model: {final_model_path}")
        return

    ks = jr.split(jr.PRNGKey(config["seed"]), 10)
    sampler = get_1d_data_sampler(
        dataset_path,
        config["data_param"]["n_eqns"],
        config["data_param"]["num_ucpair"],
    )
    batch = sampler(ks[0])
    model = DeepOSetsModel.from_data(batch, **config["model_param"], key=ks[1])
    print(f"Model parameters: {n_param(model):,}")

    if args.schedule == "cosine":
        lr = optax.cosine_decay_schedule(
            config["training_param"]["init_lr"],
            config["training_param"]["n_step"],
            alpha=1e-6,
        )
    else:
        lr = config["training_param"]["init_lr"]
    optimizer = optax.adam(lr)
    model, losses = train_model(
        model,
        sampler,
        optimizer,
        config["training_param"]["n_step"],
        key=ks[2],
        checkpoint_path=save_dir / f"deeposets_checkpoint_{args.problem}.eqx",
        save_every=args.save_every,
    )

    save_model(final_model_path, model)
    np.save(save_dir / f"losses_{args.problem}.npy", losses)
    print(f"Saved final model and losses to {save_dir}")


if __name__ == "__main__":
    main()

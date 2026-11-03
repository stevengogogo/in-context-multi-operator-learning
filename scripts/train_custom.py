"""Train DeepOSets on a user-provided custom operator dataset."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.random as jr
import numpy as np
import optax

from deeposets.configs import custom_experiment_config
from deeposets.data import get_data_sampler
from deeposets.models import DeepOSetsModel, n_param
from deeposets.training import mse_loss, normalized_mse_loss, save_model, train_model

jax.config.update("jax_enable_x64", False)


def parse_args():
    """Define and parse command-line options for custom-data training."""
    parser = argparse.ArgumentParser(description="Train DeepOSets on a custom operator dataset.")
    parser.add_argument("--train-dataset", type=Path, default=Path("data/custom/custom_dataset_train.npz"))
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/custom"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--reuse-existing", action="store_true", help="Skip training when deeposets_custom.eqx already exists.")

    parser.add_argument("--n-eqns", type=int, default=5)
    parser.add_argument("--num-ucpair", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-lr", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-embed-h", type=int, default=None)
    parser.add_argument("--n-intermediate", type=int, default=None)
    parser.add_argument("--pooling", choices=["mean", "sum", "max"], default="mean")
    parser.add_argument("--loss", choices=["mse", "normalized_mse"], default="mse")
    return parser.parse_args()


def serializable_run_config(args):
    """Return the model/training options needed to reload a custom checkpoint."""
    return {
        "train_dataset": str(args.train_dataset),
        "n_eqns": args.n_eqns,
        "num_ucpair": args.num_ucpair,
        "seed": args.seed,
        "init_lr": args.init_lr,
        "steps": args.steps,
        "width": args.width,
        "depth": args.depth,
        "n_embed_h": args.n_embed_h,
        "n_intermediate": args.n_intermediate,
        "pooling": args.pooling,
        "loss": args.loss,
    }


def main():
    """Train on a custom dataset and save the model, losses, and reload config."""
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.save_dir / "deeposets_custom.eqx"
    config_path = args.save_dir / "custom_model_config.json"
    losses_path = args.save_dir / "losses_custom.npy"

    if args.reuse_existing and model_path.exists():
        print(f"Using existing trained model: {model_path}")
        return

    config = custom_experiment_config(
        seed=args.seed,
        width=args.width,
        depth=args.depth,
        n_embed_h=args.n_embed_h,
        n_intermediate=args.n_intermediate,
        pooling=args.pooling,
        init_lr=args.init_lr,
        n_step=args.steps,
    )
    keys = jr.split(jr.PRNGKey(args.seed), 10)
    sampler = get_data_sampler(args.train_dataset, args.n_eqns, args.num_ucpair)
    batch = sampler(keys[0])
    model = DeepOSetsModel.from_data(batch, **config["model_param"], key=keys[1])
    print(f"Model parameters: {n_param(model):,}")

    lr = optax.cosine_decay_schedule(args.init_lr, args.steps, alpha=1e-6)
    optimizer = optax.adam(lr)
    loss_fn = normalized_mse_loss if args.loss == "normalized_mse" else mse_loss
    model, losses = train_model(
        model,
        sampler,
        optimizer,
        args.steps,
        key=keys[2],
        loss_fn=loss_fn,
        checkpoint_path=args.save_dir / "deeposets_custom_checkpoint.eqx",
        save_every=args.save_every,
    )

    save_model(model_path, model)
    np.save(losses_path, losses)
    config_path.write_text(json.dumps(serializable_run_config(args), indent=2), encoding="utf-8")
    print(f"Saved final model: {model_path}")
    print(f"Saved training config: {config_path}")
    print(f"Saved losses: {losses_path}")


if __name__ == "__main__":
    main()

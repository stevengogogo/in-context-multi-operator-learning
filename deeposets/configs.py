"""Default model and training configurations used by the scripts."""

import jax


def one_dim_experiment_config():
    """Defaults used for the 1D experiments in the paper code."""
    return {
        "name": "poisson_ops_deeposets",
        "wandb_mode": "disabled",
        "seed": 0,
        "data_param": {
            "n_eqns": 5,
            "n_xs": 50,
            "dx": 0.02,
            "num_ucpair": 40,
        },
        "model_param": {
            "pooling": "mean",
            "widths_net_deepset": [800, 800, 800, 800],
            "widths_net_branch": [800, 800, 800, 800, 800],
            "widths_net_trunk": [800, 800, 800, 800, 800],
            "n_embed_h": 800,
            "n_intermediate": 800,
            "activation_deepset": jax.nn.relu,
            "activation_branch": jax.nn.relu,
            "activation_trunk": jax.nn.relu,
            "final_activation_deepset": lambda x: x,
            "final_activation_branch": lambda x: x,
            "final_activation_trunk": lambda x: x,
        },
        "training_param": {
            "init_lr": 1e-4,
            "n_step": 100_000,
        },
    }


def two_dim_experiment_config():
    """Defaults used for the 2D Poisson experiment in the paper code."""
    return {
        "name": "poisson_ops_deeposets",
        "wandb_mode": "disabled",
        "seed": 0,
        "data_param": {
            "n_eqns": 5,
            "n_xs": 31,
            "dx": 0.03125,
            "num_ucpair": 40,
        },
        "model_param": {
            "pooling": "mean",
            "widths_net_deepset": [1500, 1200, 1200, 1200],
            "widths_net_branch": [1000, 1000, 800, 800, 800],
            "widths_net_trunk": [1000, 1000, 1000, 1000, 1000],
            "n_embed_h": 1200,
            "n_intermediate": 1200,
            "activation_deepset": jax.nn.relu,
            "activation_branch": jax.nn.relu,
            "activation_trunk": jax.nn.relu,
            "final_activation_deepset": lambda x: x,
            "final_activation_branch": lambda x: x,
            "final_activation_trunk": lambda x: x,
        },
        "training_param": {
            "init_lr": 1e-4,
            "n_step": 100_000,
        },
    }


def custom_experiment_config(
    seed=0,
    width=256,
    depth=4,
    n_embed_h=None,
    n_intermediate=None,
    pooling="mean",
    init_lr=1e-4,
    n_step=10_000,
):
    """Config for user-provided datasets.

    The paper experiments keep their original larger defaults above. This
    smaller, explicit default is meant as a practical starting point for custom
    data and can be overridden from the custom training script.
    """
    n_embed_h = width if n_embed_h is None else n_embed_h
    n_intermediate = width if n_intermediate is None else n_intermediate
    hidden = [width] * depth
    return {
        "name": "custom_deeposets",
        "seed": seed,
        "model_param": {
            "pooling": pooling,
            "widths_net_deepset": hidden,
            "widths_net_branch": hidden,
            "widths_net_trunk": hidden,
            "n_embed_h": n_embed_h,
            "n_intermediate": n_intermediate,
            "activation_deepset": jax.nn.relu,
            "activation_branch": jax.nn.relu,
            "activation_trunk": jax.nn.relu,
            "final_activation_deepset": lambda x: x,
            "final_activation_branch": lambda x: x,
            "final_activation_trunk": lambda x: x,
        },
        "training_param": {
            "init_lr": init_lr,
            "n_step": n_step,
        },
    }

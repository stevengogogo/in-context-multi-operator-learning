"""Evaluation utilities shared by command-line scripts and notebooks."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from deeposets.training import predict_batch


def evaluate_sampler(model, sampler, n_trials=100, seed=42):
    """Evaluate one sampler over repeated random batches."""
    keys = jax.random.split(jr.PRNGKey(seed), n_trials)
    mse_all = []
    rel_all = []

    for key in keys:
        data = sampler(key)
        pred = predict_batch(model, data).squeeze(-1)
        true = data.qoi_answer
        mse_all.append(float(jnp.mean((pred - true) ** 2)))
        rel_all.append(float(jnp.linalg.norm(pred - true) / jnp.linalg.norm(true)))

    return {
        "mse": float(np.mean(mse_all)),
        "relative_l2": float(np.mean(rel_all)),
        "mse_variance": float(np.var(mse_all)),
    }


def evaluate_context_sweep(model, sampler_factory, context_sizes, n_trials=100, seed=42):
    """Evaluate a model for several in-context dataset sizes."""
    keys = jax.random.split(jr.PRNGKey(seed), n_trials)
    mse_results = []
    rel_results = []

    for num_ucpair in context_sizes:
        sampler = sampler_factory(num_ucpair)
        mse_all = []
        rel_all = []
        for key in keys:
            data = sampler(key)
            pred = predict_batch(model, data).squeeze(-1)
            true = data.qoi_answer
            mse_all.append(float(jnp.mean((pred - true) ** 2)))
            rel_all.append(float(jnp.linalg.norm(pred - true) / jnp.linalg.norm(true)))

        mse_results.append(float(np.mean(mse_all)))
        rel_results.append(float(np.mean(rel_all)))

    return {
        "context_sizes": np.array(list(context_sizes), dtype=np.int32),
        "mse": np.array(mse_results, dtype=np.float64),
        "relative_l2": np.array(rel_results, dtype=np.float64),
    }

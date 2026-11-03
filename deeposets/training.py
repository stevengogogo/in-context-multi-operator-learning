"""Training, loss, prediction, and checkpoint helpers."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from tqdm import trange

from deeposets.data import ContextData
from deeposets.models import DeepOSetsModel


@eqx.filter_jit
def predict_batch(model: DeepOSetsModel, data: ContextData):
    """Vectorize a model over all sampled equations in a ContextData batch."""
    return jax.vmap(model, in_axes=(0, 0, 0, None))(
        data.cond_context,
        data.qoi_context,
        data.cond_query,
        data.xs_query,
    )


@eqx.filter_jit
def mse_loss(model: DeepOSetsModel, data: ContextData):
    """Mean squared error over every equation and query point in a batch."""
    qoi_pred = predict_batch(model, data)
    return jnp.mean((qoi_pred.ravel() - data.qoi_answer.ravel()) ** 2)


@eqx.filter_jit
def normalized_mse_loss(model: DeepOSetsModel, data: ContextData):
    """MSE normalized by each target field norm, with a small norm floor."""
    qoi_pred = predict_batch(model, data).squeeze(-1)
    norms = jnp.sqrt(jnp.sum(data.qoi_answer**2, axis=-1, keepdims=True))
    norms = jnp.maximum(norms, 0.5)
    return jnp.mean(((qoi_pred - data.qoi_answer) / norms) ** 2)


def train_model(
    model,
    data_sampler,
    optimizer,
    n_step,
    key,
    loss_fn=mse_loss,
    checkpoint_path=None,
    save_every=5000,
):
    """Run the optimization loop and optionally save periodic checkpoints."""

    @eqx.filter_jit
    def update_fn(model, data, state):
        loss, grad = eqx.filter_value_and_grad(loss_fn)(model, data)
        updates, new_state = optimizer.update(grad, state, model)
        return eqx.apply_updates(model, updates), new_state, loss

    state = optimizer.init(eqx.filter(model, eqx.is_array))
    losses = []
    iterator = trange(n_step)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

    for i_step in iterator:
        kd, _, key = jr.split(key, 3)
        data = data_sampler(kd)
        model, state, loss = update_fn(model, data, state)
        loss_scalar = float(loss)
        losses.append(loss_scalar)
        iterator.set_postfix({"loss": f"{loss_scalar:.6e}"})

        if checkpoint_path and (i_step + 1) % save_every == 0:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            eqx.tree_serialise_leaves(checkpoint_path, model)
            print(f"Checkpoint saved at step {i_step + 1}: {checkpoint_path}")

    return model, np.array(losses)


def save_model(path, model):
    """Serialize an Equinox model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)


def load_model(path, template_model):
    """Load an Equinox model using a same-architecture template model."""
    return eqx.tree_deserialise_leaves(path, template_model)

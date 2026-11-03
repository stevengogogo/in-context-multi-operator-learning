"""Plotting helpers for saved evaluation curves and prediction examples."""

from pathlib import Path
import os

_CACHE_DIR = Path.cwd() / ".cache"
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from deeposets.training import predict_batch


def plot_context_error(context_sizes, relative_l2, title, save_path):
    """Save the standard relative-error-versus-context-size plot."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(context_sizes, relative_l2, "ro-")
    ax.set_xlabel("Number of context examples")
    ax.set_ylabel("Relative L2 error")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_1d_prediction(model, data, problem, save_path, operator_index=0):
    """Plot a single sampled 1D operator prediction."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pred = predict_batch(model, data).squeeze(-1)
    xs = data.xs_query[:, 0]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for i in range(data.cond_context.shape[1]):
        axes[0].plot(xs, data.cond_context[operator_index, i], color="0.65", linewidth=1.0)
        axes[1].plot(xs, data.qoi_context[operator_index, i], color="0.65", linewidth=1.0)

    if problem == "poisson":
        input_label = "query forcing"
    elif problem == "burgers":
        input_label = "query initial condition"
    else:
        input_label = "query input"
    axes[0].plot(xs, data.cond_query[operator_index], color="tab:blue", linewidth=2.0, label=input_label)
    axes[1].plot(xs, data.qoi_answer[operator_index], color="black", linewidth=2.0, label="true")
    axes[1].plot(xs, pred[operator_index], color="tab:red", linestyle="--", linewidth=2.0, label="predicted")

    axes[0].set_ylabel("input")
    axes[1].set_ylabel("output")
    axes[1].set_xlabel("x")
    axes[0].legend()
    axes[1].legend()
    title = "Custom operator" if problem == "custom" else f"1D {problem.title()}"
    fig.suptitle(f"{title} prediction")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _add_image(ax, data, title, cmap, vmin=None, vmax=None):
    im = ax.imshow(data, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


def plot_2d_prediction(model, data, save_path, operator_index=0, title="2D Poisson prediction", input_title="Query f(x,y)"):
    """Plot one 2D Poisson query, prediction, truth, and error field."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pred = predict_batch(model, data).squeeze(-1)
    grid_size = int(np.sqrt(data.qoi_answer.shape[-1]))
    f_query = data.cond_query[operator_index].reshape(grid_size, grid_size)
    u_true = data.qoi_answer[operator_index].reshape(grid_size, grid_size)
    u_pred = pred[operator_index].reshape(grid_size, grid_size)
    diff = u_pred - u_true

    vmin = float(jnp.minimum(u_true.min(), u_pred.min()))
    vmax = float(jnp.maximum(u_true.max(), u_pred.max()))
    rel_l2 = float(jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    _add_image(axes[0], f_query, input_title, "RdBu")
    _add_image(axes[1], u_true, "True u(x,y)", "viridis", vmin, vmax)
    _add_image(axes[2], u_pred, "Predicted u(x,y)", "viridis", vmin, vmax)
    _add_image(axes[3], diff, "Prediction error", "RdBu", float(diff.min()), float(diff.max()))
    fig.suptitle(f"{title}, relative L2 = {rel_l2:.4f}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_custom_prediction(model, data, save_path, operator_index=0):
    """Plot a custom operator prediction using coordinates when possible."""
    coord_dim = data.xs_query.shape[-1]
    n_points = data.qoi_answer.shape[-1]
    grid_size = int(np.sqrt(n_points))

    if coord_dim == 1:
        return plot_1d_prediction(model, data, "custom", save_path, operator_index=operator_index)

    if coord_dim >= 2 and grid_size * grid_size == n_points:
        return plot_2d_prediction(
            model,
            data,
            save_path,
            operator_index=operator_index,
            title="Custom operator prediction",
            input_title="Query input",
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pred = predict_batch(model, data).squeeze(-1)

    x = np.arange(n_points)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(x, data.cond_query[operator_index], color="tab:blue", linewidth=2.0, label="query input")
    axes[1].plot(x, data.qoi_answer[operator_index], color="black", linewidth=2.0, label="true")
    axes[1].plot(x, pred[operator_index], color="tab:red", linestyle="--", linewidth=2.0, label="predicted")
    axes[0].set_ylabel("input")
    axes[1].set_ylabel("output")
    axes[1].set_xlabel("point index")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Custom operator prediction")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path

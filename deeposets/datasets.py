"""Dataset generation workflows for the paper PDE examples."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

from deeposets.solvers.one_d import (
    generate_gaussian_process,
    rbf_kernel_jax,
    solve_1d_poisson_batch,
    solve_burgers_batch,
)
from deeposets.solvers.two_d import generate_poisson2d_fields, solve_poisson2d_chunk


def _split_train_test(n_items, train_fraction):
    if n_items < 2:
        raise ValueError("At least two operators are needed to create train and test splits.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")

    n_train = int(train_fraction * n_items)
    if n_train == 0 or n_train == n_items:
        raise ValueError(
            "train_fraction produced an empty train or test split. "
            "Use more operators or choose a less extreme train_fraction."
        )
    return slice(None, n_train), slice(n_train, None)


def generate_poisson1d_dataset(
    save_dir,
    N=50,
    n_operators=200_000,
    chunk_size=100,
    n_pairs=40,
    seed=0,
    train_fraction=0.9,
):
    """Generate the 1D variable-coefficient Poisson dataset."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    xs = jnp.linspace(0.0, 1.0, N + 2)
    xs_int = xs[1:-1]
    k_all, f_all, u_all = [], [], []
    key = jax.random.PRNGKey(seed)

    for start in trange(0, n_operators, chunk_size, desc="Generating 1D Poisson"):
        count = min(chunk_size, n_operators - start)
        key, subkey = jax.random.split(key)
        ks = jax.random.split(subkey, 2)

        k_chunk = jnp.exp(
            generate_gaussian_process(
                ks[0],
                xs,
                count,
                kernel=rbf_kernel_jax,
                k_sigma=0.5,
                k_l=0.4,
            )
        )
        f_flat = generate_gaussian_process(
            ks[1],
            xs_int,
            count * n_pairs,
            kernel=rbf_kernel_jax,
            k_sigma=2.0,
            k_l=0.4,
        )
        f_chunk = f_flat.reshape(count, n_pairs, N)
        u_chunk = jax.vmap(solve_1d_poisson_batch, in_axes=(None, 0, 0))(N, k_chunk, f_chunk)

        k_all.append(np.array(k_chunk))
        f_all.append(np.array(f_chunk))
        u_all.append(np.array(u_chunk))

    k = np.concatenate(k_all, axis=0)
    f = np.concatenate(f_all, axis=0)
    u = np.concatenate(u_all, axis=0)
    train_slice, test_slice = _split_train_test(n_operators, train_fraction)

    np.savez(save_dir / "poisson1d_dataset_train.npz", k=k[train_slice], f=f[train_slice], u=u[train_slice], xs=np.array(xs))
    np.savez(save_dir / "poisson1d_dataset_test.npz", k=k[test_slice], f=f[test_slice], u=u[test_slice], xs=np.array(xs))


def generate_burgers1d_dataset(
    save_dir,
    N=100,
    M=1000,
    n_operators=200_000,
    chunk_size=100,
    n_pairs=40,
    seed=0,
    train_fraction=0.9,
):
    """Generate the 1D viscous Burgers dataset."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    xs = jnp.linspace(0.0, 1.0, N + 2)
    xs_int = xs[1:-1]
    window = jnp.sin(jnp.pi * xs_int) ** 2
    mu_all, u0_all, u_all = [], [], []
    key = jax.random.PRNGKey(seed)

    for start in trange(0, n_operators, chunk_size, desc="Generating 1D Burgers"):
        count = min(chunk_size, n_operators - start)
        key, subkey = jax.random.split(key)
        ks = jax.random.split(subkey, 2)

        log_mu = jax.random.uniform(
            ks[0],
            (count,),
            minval=jnp.log(0.001),
            maxval=jnp.log(0.1),
        )
        mu_chunk = jnp.exp(log_mu)
        u0_samples = generate_gaussian_process(
            ks[1],
            xs_int,
            count * n_pairs,
            kernel=rbf_kernel_jax,
            k_sigma=0.8,
            k_l=0.2,
        )
        u0_chunk = (u0_samples * window[None, :]).reshape(count, n_pairs, N)
        u_chunk = jax.vmap(solve_burgers_batch, in_axes=(None, None, 0, 0))(N, M, mu_chunk, u0_chunk)

        mu_all.append(np.array(mu_chunk))
        u0_all.append(np.array(u0_chunk))
        u_all.append(np.array(u_chunk))

    mu = np.concatenate(mu_all, axis=0)
    u0 = np.concatenate(u0_all, axis=0)
    u = np.concatenate(u_all, axis=0)
    train_slice, test_slice = _split_train_test(n_operators, train_fraction)

    np.savez(save_dir / "burgers1d_dataset_train.npz", mu=mu[train_slice], u0=u0[train_slice], u=u[train_slice], xs=np.array(xs))
    np.savez(save_dir / "burgers1d_dataset_test.npz", mu=mu[test_slice], u0=u0[test_slice], u=u[test_slice], xs=np.array(xs))


def generate_poisson2d_dataset(
    save_dir,
    N=32,
    n_operators=50_000,
    chunk_size=500,
    n_pairs=40,
    seed=0,
    train_fraction=0.9,
):
    """Generate the 2D variable-coefficient Poisson dataset."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_nodes = N + 1
    k_path = save_dir / "poisson2d_k_all.npy"
    f_path = save_dir / "poisson2d_f_all.npy"
    u_path = save_dir / "poisson2d_u_all.npy"

    k_all = np.lib.format.open_memmap(k_path, dtype="float32", mode="w+", shape=(n_operators, n_nodes, n_nodes))
    f_all = np.lib.format.open_memmap(f_path, dtype="float32", mode="w+", shape=(n_operators, n_pairs, n_nodes, n_nodes))
    u_all = np.lib.format.open_memmap(u_path, dtype="float32", mode="w+", shape=(n_operators, n_pairs, n_nodes, n_nodes))

    key = jax.random.PRNGKey(seed)
    for start in trange(0, n_operators, chunk_size, desc="Generating 2D Poisson"):
        count = min(chunk_size, n_operators - start)
        key, subkey = jax.random.split(key)

        k_chunk, f_chunk = generate_poisson2d_fields(subkey, N, count, n_pairs)
        u_chunk = solve_poisson2d_chunk(N, k_chunk, f_chunk)
        u_chunk.block_until_ready()

        stop = start + count
        k_all[start:stop] = np.array(k_chunk, dtype=np.float32)
        f_all[start:stop] = np.array(f_chunk, dtype=np.float32)
        u_all[start:stop] = np.array(u_chunk, dtype=np.float32)

        if (start // chunk_size + 1) % 50 == 0:
            jax.clear_caches()

    k_all.flush()
    f_all.flush()
    u_all.flush()

    xs = np.array(jnp.linspace(0.0, 1.0, N + 1))
    train_slice, test_slice = _split_train_test(n_operators, train_fraction)
    np.savez(save_dir / "poisson2d_dataset_train.npz", k=k_all[train_slice], f=f_all[train_slice], u=u_all[train_slice], xs=xs)
    np.savez(save_dir / "poisson2d_dataset_test.npz", k=k_all[test_slice], f=f_all[test_slice], u=u_all[test_slice], xs=xs)

    # The memory-mapped intermediates duplicate the final files; remove them to save disk.
    del k_all, f_all, u_all
    for path in (k_path, f_path, u_path):
        path.unlink()

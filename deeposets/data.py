"""Dataset loading, sampling, and custom-data conversion utilities."""

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class ContextData(NamedTuple):
    """One in-context operator-learning batch.

    Shapes:
      cond_context: (n_eqns, num_ucpair - 1, n_points)
      qoi_context:  (n_eqns, num_ucpair - 1, n_points)
      cond_query:   (n_eqns, n_points)
      xs_query:     (n_points, coordinate_dim)
      qoi_answer:   (n_eqns, n_points)

    Each sampled equation/operator receives several context input-output
    examples plus one query input whose output is predicted by the model.
    """

    cond_context: jnp.ndarray
    qoi_context: jnp.ndarray
    cond_query: jnp.ndarray
    xs_query: jnp.ndarray
    xs_cond: jnp.ndarray
    qoi_answer: jnp.ndarray


class DatasetInfo(NamedTuple):
    """Small summary of an on-disk dataset."""

    path: Path
    problem: str
    split: str
    n_operators: int
    n_pairs: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


def infer_dataset_problem(dataset_path):
    """Infer the supported experiment name from a dataset filename."""
    name = Path(dataset_path).name.lower()
    if "poisson1d" in name:
        return "poisson1d"
    if "burgers1d" in name:
        return "burgers1d"
    if "poisson2d" in name:
        return "poisson2d"
    raise ValueError(f"Unrecognized dataset path: {dataset_path}")


def detect_dataset_problem(dataset_path):
    """Infer the dataset type from standard keys first, then from the filename."""
    data = np.load(Path(dataset_path), mmap_mode="r")
    if {"cond", "qoi"}.issubset(data.files):
        return "custom"
    return infer_dataset_problem(dataset_path)


def infer_dataset_split(dataset_path):
    """Infer whether a dataset path points to a train or test split."""
    name = Path(dataset_path).name.lower()
    if "train" in name:
        return "train"
    if "test" in name:
        return "test"
    return "unknown"


def load_dataset(dataset_path, mmap_mode="r"):
    """Open a generated ``.npz`` dataset without copying all arrays into memory."""
    return np.load(Path(dataset_path), mmap_mode=mmap_mode)


def describe_dataset(dataset_path):
    """Return dataset metadata that is useful for scripts and notebooks."""
    dataset_path = Path(dataset_path)
    data = load_dataset(dataset_path)
    problem = detect_dataset_problem(dataset_path)

    if problem == "custom":
        inputs = data["cond"]
        outputs = data["qoi"]
    elif problem == "poisson1d":
        inputs = data["f"]
        outputs = data["u"]
    elif problem == "burgers1d":
        inputs = data["u0"]
        outputs = data["u"]
    else:
        inputs = data["f"]
        outputs = data["u"]

    return DatasetInfo(
        path=dataset_path,
        problem=problem,
        split=infer_dataset_split(dataset_path),
        n_operators=int(inputs.shape[0]),
        n_pairs=int(inputs.shape[1]),
        input_shape=tuple(inputs.shape[2:]),
        output_shape=tuple(outputs.shape[2:]),
    )


def concatenate_data(datasets):
    """Concatenate multiple task batches while keeping coordinate arrays shared."""
    datasets = tuple(datasets)
    if len(datasets) == 1:
        return datasets[0]

    return ContextData(
        cond_context=jnp.concatenate([d.cond_context for d in datasets], axis=0),
        qoi_context=jnp.concatenate([d.qoi_context for d in datasets], axis=0),
        cond_query=jnp.concatenate([d.cond_query for d in datasets], axis=0),
        xs_query=datasets[0].xs_query,
        xs_cond=datasets[0].xs_cond,
        qoi_answer=jnp.concatenate([d.qoi_answer for d in datasets], axis=0),
    )


def get_data_sampler(dataset_path, n_eqns, num_ucpair):
    """Dispatch to the correct sampler for a generated dataset path."""
    problem = detect_dataset_problem(dataset_path)
    if problem == "custom":
        return get_custom_data_sampler(dataset_path, n_eqns, num_ucpair)
    if problem in {"poisson1d", "burgers1d"}:
        return get_1d_data_sampler(dataset_path, n_eqns, num_ucpair)
    if problem == "poisson2d":
        return get_2d_data_sampler(dataset_path, n_eqns, num_ucpair)
    raise ValueError(f"Unsupported dataset problem: {problem}")


def _standardize_coords(xs, n_points, spatial_shape):
    if xs is None:
        if len(spatial_shape) == 2:
            x = np.linspace(0.0, 1.0, spatial_shape[0])
            y = np.linspace(0.0, 1.0, spatial_shape[1])
            xx, yy = np.meshgrid(x, y, indexing="ij")
            return np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
        return np.linspace(0.0, 1.0, n_points, dtype=np.float32).reshape(-1, 1)

    xs = np.asarray(xs)
    if xs.ndim == 1:
        if xs.shape[0] != n_points:
            raise ValueError(f"1D coordinates must have length {n_points}; got {xs.shape[0]}")
        return xs.reshape(-1, 1).astype(np.float32)
    if xs.ndim == 2 and xs.shape[0] == n_points:
        return xs.astype(np.float32)
    if xs.ndim >= 2 and xs.shape[:-1] == spatial_shape and xs.shape[-1] in {1, 2, 3}:
        return xs.reshape(n_points, xs.shape[-1]).astype(np.float32)

    raise ValueError(
        "Coordinates must be shape (n_points,), (n_points, coord_dim), "
        "or spatial_shape + (coord_dim,)."
    )


def standardize_operator_arrays(cond, qoi, xs=None):
    """Convert user arrays to the DeepOSets custom dataset convention.

    ``cond`` and ``qoi`` must have shape ``(n_operators, n_pairs, ...)``.
    The trailing spatial dimensions are flattened for training. The current
    The current model expects the condition and solution fields to live on the
    same number of query points.
    """
    cond = np.asarray(cond, dtype=np.float32)
    qoi = np.asarray(qoi, dtype=np.float32)
    if cond.ndim < 3 or qoi.ndim < 3:
        raise ValueError("cond and qoi must have shape (n_operators, n_pairs, ...)")
    if cond.shape[:2] != qoi.shape[:2]:
        raise ValueError(f"cond and qoi must share the first two axes; got {cond.shape[:2]} and {qoi.shape[:2]}")

    n_operators, n_pairs = cond.shape[:2]
    cond_flat = cond.reshape(n_operators, n_pairs, -1)
    qoi_flat = qoi.reshape(n_operators, n_pairs, -1)
    if cond_flat.shape[-1] != qoi_flat.shape[-1]:
        raise ValueError(
            "cond and qoi must have the same flattened spatial size. "
            f"Got {cond_flat.shape[-1]} and {qoi_flat.shape[-1]}."
        )

    n_points = cond_flat.shape[-1]
    spatial_shape = tuple(qoi.shape[2:])
    xs_flat = _standardize_coords(xs, n_points, spatial_shape)
    if xs_flat.shape[0] != n_points:
        raise ValueError(f"Coordinates describe {xs_flat.shape[0]} points, but fields have {n_points} points.")

    return cond_flat, qoi_flat, xs_flat, spatial_shape


def save_custom_operator_dataset(
    save_dir,
    cond,
    qoi,
    xs=None,
    train_fraction=0.9,
    seed=0,
    shuffle=False,
    prefix="custom",
):
    """Save a train/test split for user-provided operator data.

    The saved files use keys ``cond``, ``qoi``, ``xs``, and ``spatial_shape``.
    This is the generic format consumed by the custom training/evaluation
    scripts and by ``get_custom_data_sampler``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cond_flat, qoi_flat, xs_flat, spatial_shape = standardize_operator_arrays(cond, qoi, xs)

    n_operators = cond_flat.shape[0]
    if n_operators < 2:
        raise ValueError("At least two operators are needed to create train and test splits.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")

    indices = np.arange(n_operators)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    n_train = int(train_fraction * n_operators)
    if n_train == 0 or n_train == n_operators:
        raise ValueError(
            "train_fraction produced an empty train or test split. "
            "Use more operators or choose a less extreme train_fraction."
        )
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    np.savez(
        save_dir / f"{prefix}_dataset_train.npz",
        cond=cond_flat[train_indices],
        qoi=qoi_flat[train_indices],
        xs=xs_flat,
        spatial_shape=np.array(spatial_shape, dtype=np.int32),
    )
    np.savez(
        save_dir / f"{prefix}_dataset_test.npz",
        cond=cond_flat[test_indices],
        qoi=qoi_flat[test_indices],
        xs=xs_flat,
        spatial_shape=np.array(spatial_shape, dtype=np.int32),
    )
    return save_dir / f"{prefix}_dataset_train.npz", save_dir / f"{prefix}_dataset_test.npz"


def prepare_custom_dataset_from_npz(
    source_path,
    save_dir,
    input_key="cond",
    output_key="qoi",
    coords_key="xs",
    train_fraction=0.9,
    seed=0,
    shuffle=False,
    prefix="custom",
):
    """Load user arrays from an ``.npz`` file and save the standard custom split."""
    source = np.load(Path(source_path), mmap_mode="r")
    if input_key not in source.files:
        raise KeyError(f"Input key '{input_key}' not found in {source_path}. Available keys: {source.files}")
    if output_key not in source.files:
        raise KeyError(f"Output key '{output_key}' not found in {source_path}. Available keys: {source.files}")
    xs = source[coords_key] if coords_key and coords_key in source.files else None
    return save_custom_operator_dataset(
        save_dir=save_dir,
        cond=source[input_key],
        qoi=source[output_key],
        xs=xs,
        train_fraction=train_fraction,
        seed=seed,
        shuffle=shuffle,
        prefix=prefix,
    )


def get_1d_data_sampler(dataset_path, n_eqns, num_ucpair):
    """Load a 1D dataset and return a callable that samples ContextData batches."""
    dataset_path = Path(dataset_path)
    data = np.load(dataset_path, mmap_mode="r")
    path_name = dataset_path.name.lower()

    if "burgers1d" in path_name:
        u0_np = data["u0"]
        u_np = data["u"]
        xs_1d = data["xs"]
        n_operators, n_pairs, n_xs = u_np.shape
        xs = jnp.array(xs_1d[1:-1]).reshape(-1, 1)

        print(f"Loaded Burgers 1D dataset: {n_operators} operators, {n_pairs} pairs, N={n_xs}")

        def data_sampler(key):
            ks = jax.random.split(key, 2)
            op_idx = np.array(jax.random.randint(ks[0], (n_eqns,), 0, n_operators))
            pair_idx = np.array(jax.random.randint(ks[1], (n_eqns, num_ucpair), 0, n_pairs))
            u0_batch = u0_np[op_idx[:, None], pair_idx]
            u_batch = u_np[op_idx[:, None], pair_idx]
            return ContextData(
                cond_context=jnp.array(u0_batch[:, :-1, :]),
                qoi_context=jnp.array(u_batch[:, :-1, :]),
                cond_query=jnp.array(u0_batch[:, -1, :]),
                xs_query=xs,
                xs_cond=xs,
                qoi_answer=jnp.array(u_batch[:, -1, :]),
            )

        return data_sampler

    if "poisson1d" in path_name:
        f_np = data["f"]
        u_np = data["u"][:, :, 1:-1]
        xs_1d = data["xs"]
        n_operators, n_pairs, n_xs = f_np.shape
        xs = jnp.array(xs_1d[1:-1]).reshape(-1, 1)

        print(f"Loaded Poisson 1D dataset: {n_operators} operators, {n_pairs} pairs, N={n_xs}")

        def data_sampler(key):
            ks = jax.random.split(key, 2)
            op_idx = np.array(jax.random.randint(ks[0], (n_eqns,), 0, n_operators))
            pair_idx = np.array(jax.random.randint(ks[1], (n_eqns, num_ucpair), 0, n_pairs))
            f_batch = f_np[op_idx[:, None], pair_idx]
            u_batch = u_np[op_idx[:, None], pair_idx]
            return ContextData(
                cond_context=jnp.array(f_batch[:, :-1, :]),
                qoi_context=jnp.array(u_batch[:, :-1, :]),
                cond_query=jnp.array(f_batch[:, -1, :]),
                xs_query=xs,
                xs_cond=xs,
                qoi_answer=jnp.array(u_batch[:, -1, :]),
            )

        return data_sampler

    raise ValueError(f"Unrecognized 1D dataset path: {dataset_path}")


def get_2d_data_sampler(dataset_path, n_eqns, num_ucpair):
    """Load a 2D Poisson dataset and return a ContextData sampler."""
    dataset_path = Path(dataset_path)
    data = np.load(dataset_path, mmap_mode="r")
    if "poisson2d" not in dataset_path.name.lower():
        raise ValueError(f"Unrecognized 2D dataset path: {dataset_path}")

    f_np = data["f"]
    u_np = data["u"]
    xs_1d = data["xs"]

    n_operators, n_pairs, Np1, _ = u_np.shape
    N = Np1 - 1
    xs_int = xs_1d[1:-1]
    XX, YY = np.meshgrid(xs_int, xs_int, indexing="ij")
    xs_interior = jnp.array(np.stack([XX.ravel(), YY.ravel()], axis=-1))

    print(f"Loaded Poisson 2D dataset: {n_operators} operators, {n_pairs} pairs, N={N}")

    def data_sampler(key):
        ks = jax.random.split(key, 2)
        op_idx = np.array(jax.random.randint(ks[0], (n_eqns,), 0, n_operators))
        pair_idx = np.array(jax.random.randint(ks[1], (n_eqns, num_ucpair), 0, n_pairs))

        f_batch = f_np[op_idx[:, None], pair_idx][:, :, 1:-1, 1:-1]
        u_batch = u_np[op_idx[:, None], pair_idx][:, :, 1:-1, 1:-1]

        n_int = (N - 1) ** 2
        f_batch = f_batch.reshape(n_eqns, num_ucpair, n_int)
        u_batch = u_batch.reshape(n_eqns, num_ucpair, n_int)

        return ContextData(
            cond_context=jnp.array(f_batch[:, :-1, :]),
            qoi_context=jnp.array(u_batch[:, :-1, :]),
            cond_query=jnp.array(f_batch[:, -1, :]),
            xs_query=xs_interior,
            xs_cond=xs_interior,
            qoi_answer=jnp.array(u_batch[:, -1, :]),
        )

    return data_sampler


def get_custom_data_sampler(dataset_path, n_eqns, num_ucpair):
    """Load a user-provided custom operator dataset.

    Custom datasets use ``cond`` for the input/query field, ``qoi`` for the
    target/output field, and ``xs`` for coordinates. Arrays are expected to be
    flattened to ``(n_operators, n_pairs, n_points)`` by
    ``save_custom_operator_dataset`` or ``prepare_custom_dataset_from_npz``.
    """
    dataset_path = Path(dataset_path)
    data = np.load(dataset_path, mmap_mode="r")
    if not {"cond", "qoi", "xs"}.issubset(data.files):
        raise ValueError(f"Custom dataset must contain cond, qoi, and xs arrays: {dataset_path}")

    cond_np = data["cond"]
    qoi_np = data["qoi"]
    xs = jnp.array(data["xs"])
    n_operators, n_pairs, n_points = cond_np.shape
    if qoi_np.shape != cond_np.shape:
        raise ValueError(f"cond and qoi must have matching shapes; got {cond_np.shape} and {qoi_np.shape}")

    print(f"Loaded custom dataset: {n_operators} operators, {n_pairs} pairs, {n_points} points")

    def data_sampler(key):
        ks = jax.random.split(key, 2)
        op_idx = np.array(jax.random.randint(ks[0], (n_eqns,), 0, n_operators))
        pair_idx = np.array(jax.random.randint(ks[1], (n_eqns, num_ucpair), 0, n_pairs))
        cond_batch = cond_np[op_idx[:, None], pair_idx]
        qoi_batch = qoi_np[op_idx[:, None], pair_idx]
        return ContextData(
            cond_context=jnp.array(cond_batch[:, :-1, :]),
            qoi_context=jnp.array(qoi_batch[:, :-1, :]),
            cond_query=jnp.array(cond_batch[:, -1, :]),
            xs_query=xs,
            xs_cond=xs,
            qoi_answer=jnp.array(qoi_batch[:, -1, :]),
        )

    return data_sampler

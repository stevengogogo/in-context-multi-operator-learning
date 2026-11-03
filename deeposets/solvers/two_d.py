"""JAX solvers and random field generators for the 2D Poisson example."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def solve_2d_poisson_fem(N, k, f):
    """Finite-element solve for -div(k grad u) = f on [0, 1]^2."""
    h = 1.0 / N
    n_nodes = (N + 1) ** 2
    n_int = (N - 1) ** 2
    det_J = (h / 2.0) ** 2

    ei_np = np.repeat(np.arange(N), N)
    ej_np = np.tile(np.arange(N), N)
    n0_np = ei_np * (N + 1) + ej_np
    n1_np = ei_np * (N + 1) + ej_np + 1
    n2_np = (ei_np + 1) * (N + 1) + ej_np + 1
    n3_np = (ei_np + 1) * (N + 1) + ej_np
    elem_nodes_np = np.stack([n0_np, n1_np, n2_np, n3_np], axis=1)

    interior_np = np.array([i * (N + 1) + j for i in range(1, N) for j in range(1, N)])
    interior_set_np = np.zeros(n_nodes, dtype=bool)
    interior_set_np[interior_np] = True

    rows_np = np.repeat(elem_nodes_np, 4, axis=1).ravel()
    cols_np = np.tile(elem_nodes_np, (1, 4)).ravel()
    mask_np = interior_set_np[rows_np] & interior_set_np[cols_np]

    g2l_np = np.zeros(n_nodes, dtype=int)
    g2l_np[interior_np] = np.arange(n_int)

    elem_nodes = jnp.array(elem_nodes_np)
    interior = jnp.array(interior_np)
    rows_loc = jnp.array(g2l_np[rows_np[mask_np]])
    cols_loc = jnp.array(g2l_np[cols_np[mask_np]])
    mask_indices = jnp.array(np.where(mask_np)[0])

    gp = 1.0 / jnp.sqrt(3.0)
    gauss_pts = jnp.array([[-gp, -gp], [-gp, gp], [gp, -gp], [gp, gp]])
    gauss_wts = jnp.ones(4)

    def shape_vals(xi, eta):
        return jnp.array(
            [
                0.25 * (1 - xi) * (1 - eta),
                0.25 * (1 + xi) * (1 - eta),
                0.25 * (1 + xi) * (1 + eta),
                0.25 * (1 - xi) * (1 + eta),
            ]
        )

    def shape_grads(xi, eta):
        dN_dxi = jnp.array(
            [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)]
        )
        dN_deta = jnp.array(
            [-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)]
        )
        return (2 / h) * dN_dxi, (2 / h) * dN_deta

    def elem_matrices(k_e, f_e):
        def gauss_contrib(gpt, w):
            xi, eta = gpt[0], gpt[1]
            N_vals = shape_vals(xi, eta)
            dN_dx, dN_dy = shape_grads(xi, eta)
            k_h = jnp.dot(N_vals, k_e)
            f_h = jnp.dot(N_vals, f_e)
            K_c = w * k_h * det_J * (jnp.outer(dN_dx, dN_dx) + jnp.outer(dN_dy, dN_dy))
            F_c = w * f_h * det_J * N_vals
            return K_c, F_c

        K_cs, F_cs = jax.vmap(gauss_contrib)(gauss_pts, gauss_wts)
        return jnp.sum(K_cs, axis=0), jnp.sum(F_cs, axis=0)

    k_elems = k.ravel()[elem_nodes]
    f_elems = f.ravel()[elem_nodes]
    K_elems, F_elems = jax.vmap(elem_matrices)(k_elems, f_elems)

    F_global = jnp.zeros(n_nodes).at[elem_nodes.ravel()].add(F_elems.ravel())
    F_int = F_global[interior]
    K_vals_int = K_elems.ravel()[mask_indices]

    def matvec(v):
        return jnp.zeros(n_int).at[rows_loc].add(K_vals_int * v[cols_loc])

    u_int, _ = jax.scipy.sparse.linalg.cg(matvec, F_int, tol=1e-6, maxiter=2000)
    return jnp.zeros(n_nodes).at[interior].set(u_int).reshape(N + 1, N + 1)


solve_2d_poisson_fem_batch = jax.jit(
    jax.vmap(solve_2d_poisson_fem, in_axes=(None, None, 0)),
    static_argnums=(0,),
)


@partial(jax.jit, static_argnames=("num", "n_features"))
def sample_random_field_2d(key, xs, num, sigma, l, n_features=100):
    """Approximate an RBF Gaussian random field with random Fourier features."""
    ks = jax.random.split(key, 3)
    omega = jax.random.normal(ks[0], (n_features, 2)) / l
    phi = jax.random.uniform(ks[1], (n_features,)) * 2 * jnp.pi
    features = jnp.cos(xs @ omega.T + phi)
    coeffs = jax.random.normal(ks[2], (num, n_features)) * sigma * jnp.sqrt(2.0 / n_features)
    return coeffs @ features.T


@partial(jax.jit, static_argnames=("N", "n_operators", "n_pairs"))
def generate_poisson2d_fields(key, N, n_operators, n_pairs, k_sigma=0.3, k_l=0.4, f_sigma=5.0, f_l=0.4):
    """Sample coefficient and forcing fields for the 2D Poisson dataset."""
    xs_1d = jnp.linspace(0.0, 1.0, N + 1)
    XX, YY = jnp.meshgrid(xs_1d, xs_1d, indexing="ij")
    xs_full = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)

    ks = jax.random.split(key, 2)
    k_flat = jnp.exp(sample_random_field_2d(ks[0], xs_full, n_operators, sigma=k_sigma, l=k_l))
    f_flat = sample_random_field_2d(ks[1], xs_full, n_operators * n_pairs, sigma=f_sigma, l=f_l)
    return (
        k_flat.reshape(n_operators, N + 1, N + 1),
        f_flat.reshape(n_operators, n_pairs, N + 1, N + 1),
    )


@partial(jax.jit, static_argnames=("N",))
def solve_poisson2d_chunk(N, k_chunk, f_chunk):
    """Solve a chunk of 2D Poisson operators and right-hand sides."""
    def solve_one(k, fs):
        return jax.vmap(solve_2d_poisson_fem, in_axes=(None, None, 0))(N, k, fs)

    return jax.vmap(solve_one)(k_chunk, f_chunk)

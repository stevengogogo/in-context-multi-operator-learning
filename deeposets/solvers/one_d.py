"""JAX numerical solvers and random-function utilities for 1D examples."""

from functools import partial

from einshape import jax_einshape as einshape
import jax
from jax import lax
import jax.numpy as jnp


@jax.jit
def tridiagonal_solve(dl, d, du, b):
    """Thomas algorithm implemented with JAX primitives."""
    prepend_zero = lambda x: jnp.append(jnp.zeros([1], dtype=x.dtype), x[:-1])
    fwd1 = lambda tu_, x: x[1] / (x[0] - x[2] * tu_)
    fwd2 = lambda b_, x: (x[0] - x[3] * b_) / (x[1] - x[3] * x[2])
    bwd1 = lambda x_, x: x[0] - x[1] * x_
    double = lambda f, args: (f(*args), f(*args))

    _, tu_ = lax.scan(
        lambda tu_, x: double(fwd1, (tu_, x)),
        du[0] / d[0],
        (d, du, dl),
        unroll=32,
    )
    _, b_ = lax.scan(
        lambda b_, x: double(fwd2, (b_, x)),
        b[0] / d[0],
        (b, d, prepend_zero(tu_), dl),
        unroll=32,
    )
    _, x_ = lax.scan(
        lambda x_, x: double(bwd1, (x_, x)),
        b_[-1],
        (b_[::-1], tu_[::-1]),
        unroll=32,
    )
    return x_[::-1]


@jax.jit
def laplace_u(u, dx):
    """Second derivative helper used by legacy experiments."""
    u = u / dx
    uxx = (u[:-2] + u[2:] - 2 * u[1:-1]) / dx
    uxx_left = (2 * u[0] - 5 * u[1] + 4 * u[2] - u[3]) / dx
    uxx_right = (2 * u[-1] - 5 * u[-2] + 4 * u[-3] - u[-4]) / dx
    return jnp.pad(uxx, (1, 1), mode="constant", constant_values=(uxx_left, uxx_right))


@partial(jax.jit, static_argnames=("N",))
def solve_1d_poisson(N, k, f):
    """Solve -(k(x) u_x)_x = f on [0, 1] with homogeneous Dirichlet BCs."""
    h = 1.0 / (N + 1)
    k_half_plus = 0.5 * (k[1:-1] + k[2:])
    k_half_minus = 0.5 * (k[:-2] + k[1:-1])

    d = (k_half_minus + k_half_plus) / h**2
    du = jnp.append(-k_half_plus[:-1] / h**2, 0.0)
    dl = jnp.append(0.0, -k_half_minus[1:] / h**2)

    out_u = tridiagonal_solve(dl, d, du, f)
    return jnp.pad(out_u, (1, 1), mode="constant", constant_values=0.0)


solve_1d_poisson_batch = jax.jit(
    jax.vmap(solve_1d_poisson, in_axes=(None, None, 0)),
    static_argnums=(0,),
)


@partial(jax.jit, static_argnames=("N", "M"))
def solve_burgers(N, M, mu, u0):
    """IMEX solver for u_t + u u_x = mu u_xx with zero boundary conditions."""
    dx = 1.0 / (N + 1)
    dt = 1.0 / M
    alpha = dt * mu / dx**2

    d = jnp.ones(N) * (1.0 + 2.0 * alpha)
    dl = jnp.append(0.0, -alpha * jnp.ones(N - 1))
    du = jnp.append(-alpha * jnp.ones(N - 1), 0.0)

    def step(u, _):
        u_full = jnp.pad(u, (1, 1), mode="constant", constant_values=0.0)
        u_x_forward = (u_full[2:] - u_full[1:-1]) / dx
        u_x_backward = (u_full[1:-1] - u_full[:-2]) / dx
        u_x = jnp.where(u >= 0, u_x_backward, u_x_forward)
        rhs = u - dt * u * u_x
        return tridiagonal_solve(dl, d, du, rhs), None

    u_final, _ = lax.scan(step, u0, None, length=M)
    return u_final


solve_burgers_batch = jax.jit(
    jax.vmap(solve_burgers, in_axes=(None, None, None, 0)),
    static_argnums=(0, 1),
)


@partial(jax.jit, static_argnames=("N",))
def solve_porous(L, N, u_left, u_right, a, k, c):
    """Solve -a u_xx + k(x) u = c with Dirichlet boundary conditions."""
    dx = L / N
    du = -a * jnp.array([1.0] * (N - 2) + [0.0])
    dl = -a * jnp.array([0.0] + [1.0] * (N - 2))
    d = (2.0 * a) * jnp.ones((N - 1,)) + k * dx * dx

    b = c * dx * dx * jnp.ones((N - 1,))
    b = b.at[0].add(a * u_left)
    b = b.at[-1].add(a * u_right)

    out_u = tridiagonal_solve(dl, d, du, b)
    return jnp.pad(out_u, (1, 1), mode="constant", constant_values=(u_left, u_right))


solve_porous_batch = jax.jit(
    jax.vmap(solve_porous, in_axes=(None, None, None, None, None, 0, None)),
    static_argnums=(1,),
)


@partial(jax.jit, static_argnames=("N",))
def solve_poisson(L, N, u_left, u_right, c):
    """Solve u_xx = c on [0, L] with Dirichlet boundary conditions."""
    dx = L / N
    du = jnp.array([1.0] * (N - 2) + [0.0])
    dl = jnp.array([0.0] + [1.0] * (N - 2))
    d = -2.0 * jnp.ones((N - 1,))

    b = c * dx * dx
    b = b.at[0].add(-u_left)
    b = b.at[-1].add(-u_right)

    out_u = tridiagonal_solve(dl, d, du, b)
    return jnp.pad(out_u, (1, 1), mode="constant", constant_values=(u_left, u_right))


solve_poisson_batch = jax.jit(
    jax.vmap(solve_poisson, in_axes=(None, None, None, None, 0)),
    static_argnums=(1,),
)


@jax.jit
def solve_ode_auto_const_fn(dt, coeff_a, coeff_b, init, control):
    """Euler solve for u_dot = a c(t) + b."""
    rhs = lambda c, u: coeff_a * c + coeff_b
    _, us = jax.lax.scan(lambda u, c: (u + dt * rhs(c, u), u), init, control)
    return us


solve_ode_auto_const_fn_batch = jax.jit(
    jax.vmap(solve_ode_auto_const_fn, in_axes=(None, None, None, None, 0)),
    static_argnums=(0,),
)


@jax.jit
def solve_ode_auto_linear1_fn(dt, coeff_a, coeff_b, init, control):
    """Euler solve for u_dot = a c(t) u + b."""
    rhs = lambda c, u: coeff_a * c * u + coeff_b
    _, us = jax.lax.scan(lambda u, c: (u + dt * rhs(c, u), u), init, control)
    return us


solve_ode_auto_linear1_fn_batch = jax.jit(
    jax.vmap(solve_ode_auto_linear1_fn, in_axes=(None, None, None, None, 0)),
    static_argnums=(0,),
)


@jax.jit
def solve_ode_auto_linear2_fn(dt, coeff_a1, coeff_a2, coeff_a3, init, control):
    """Euler solve for u_dot = a1 u + a2 c(t) + a3."""
    rhs = lambda c, u: coeff_a1 * u + coeff_a2 * c + coeff_a3
    _, us = jax.lax.scan(lambda u, c: (u + dt * rhs(c, u), u), init, control)
    return us


solve_ode_auto_linear2_fn_batch = jax.jit(
    jax.vmap(solve_ode_auto_linear2_fn, in_axes=(None, None, None, None, None, 0)),
    static_argnums=(0,),
)


def generate_sin(xs, amp, period, phase):
    """Generate a sinusoid on coordinates `xs`."""
    return amp * jnp.sin(xs * 2 * jnp.pi / period + phase)


generate_sin_batch = jax.jit(jax.vmap(generate_sin, [None, 0, 0, 0], 0))


def solve_damped_oscillator(xs, amp, period, phase, decay):
    """Generate a decaying sinusoidal trajectory."""
    return generate_sin(xs, amp, period, phase) * jnp.exp(-decay * xs)


solve_damped_oscillator_batch = jax.jit(
    jax.vmap(solve_damped_oscillator, [None, 0, 0, 0, None], 0)
)


@partial(jax.jit, static_argnames=("num", "kernel"))
def generate_gaussian_process(key, ts, num, kernel, k_sigma, k_l):
    """Sample `num` draws from a zero-mean GP on the grid `ts`."""
    length = len(ts)
    mean = jnp.zeros((num, length))
    cov = kernel(ts, ts, sigma=k_sigma, l=k_l)
    cov = einshape("ii->nii", cov, n=num)
    return jax.random.multivariate_normal(key, mean=mean, cov=cov, shape=(num,), method="svd")


def rbf_kernel_jax(x1, x2, sigma, l):
    """Squared-exponential covariance matrix on two 1D grids."""
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing="ij")
    sq_norm = (xx1 - xx2) ** 2 / (l**2)
    return sigma**2 * jnp.exp(-0.5 * sq_norm)

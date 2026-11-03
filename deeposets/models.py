"""Neural network components for the DeepOSets experiments.

The model has three pieces:
  1. A DeepSet encoder that summarizes context input-output examples.
  2. A branch network that combines that summary with the query input field.
  3. A trunk network that evaluates the predicted output at coordinates.
"""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from deeposets.data import ContextData


def n_param(tree):
    """Count trainable array parameters in an Equinox model."""
    return len(jax.flatten_util.ravel_pytree(eqx.filter(tree, eqx.is_array))[0])


class MLP(eqx.Module):
    """Plain feed-forward network used inside the DeepSet and DeepONet blocks."""

    layers: list[eqx.nn.Linear]
    activation: Callable[[jnp.ndarray], jnp.ndarray]
    final_activation: Callable[[jnp.ndarray], jnp.ndarray]

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_sizes: list[int],
        activation=jax.nn.relu,
        final_activation=lambda x: x,
        key=jr.PRNGKey(0),
    ):
        k_layers = jax.random.split(key, len(width_sizes) + 1)
        self.activation = activation
        self.final_activation = final_activation
        self.layers = []

        if width_sizes:
            self.layers.append(eqx.nn.Linear(in_size, width_sizes[0], key=k_layers[0], dtype=jnp.float32))
            for i, (w_in, w_out) in enumerate(zip(width_sizes[:-1], width_sizes[1:])):
                self.layers.append(eqx.nn.Linear(w_in, w_out, key=k_layers[i + 1]))
            self.layers.append(eqx.nn.Linear(width_sizes[-1], out_size, key=k_layers[-1]))
        else:
            self.layers.append(eqx.nn.Linear(in_size, out_size, key=k_layers[0]))

    def __call__(self, x, key=jr.PRNGKey(0)):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.final_activation(self.layers[-1](x))


class DeepSet(eqx.Module):
    """Permutation-invariant encoder for the context input-output examples."""

    net: eqx.Module
    pooling: Callable[[jnp.ndarray], jnp.ndarray]

    def __init__(self, net, pooling="mean", key=jr.PRNGKey(0)):
        self.net = net
        if pooling == "mean":
            self.pooling = lambda x: jnp.mean(x, axis=0)
        elif pooling == "max":
            self.pooling = lambda x: jnp.max(x, axis=0)
        elif pooling == "sum":
            self.pooling = lambda x: jnp.sum(x, axis=0)
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")

    def __call__(self, x: jnp.ndarray, key=jr.PRNGKey(0)):
        return self.pooling(jax.vmap(self.net)(x))


class DeepONet(eqx.Module):
    """Branch-trunk operator network evaluated at query coordinates."""

    net_branch: eqx.Module
    net_trunk: eqx.Module
    bias: jnp.ndarray

    def __init__(self, net_branch, net_trunk, key=jr.PRNGKey(0)):
        self.net_branch = net_branch
        self.net_trunk = net_trunk
        self.bias = jr.normal(key, (1,))

    @eqx.filter_jit
    def __call__(self, x_branch: jnp.ndarray, x_trunk: jnp.ndarray, key=jr.PRNGKey(0)):
        return jnp.dot(self.net_branch(x_branch), self.net_trunk(x_trunk)) + self.bias

    def get_net_with_fixed_condition(self, x_branch: jnp.ndarray):
        out_branch = self.net_branch(x_branch)

        def deeponet_with_fixed_branch(x_trunk, key=jr.PRNGKey(0)):
            return jnp.dot(out_branch, self.net_trunk(x_trunk)) + self.bias

        return deeponet_with_fixed_branch


class DeepOSetsModel(eqx.Module):
    """DeepOSets model used in the experiments.

    The context pairs are encoded by a DeepSet. That embedding is concatenated
    with the query condition/forcing and passed through the DeepONet branch.
    The trunk network evaluates the predicted quantity of interest at each
    coordinate in `xs_query`.
    """

    setnet: DeepSet
    deeponet: DeepONet

    @classmethod
    def from_data(
        cls,
        data: ContextData,
        pooling: str,
        widths_net_deepset: list[int],
        widths_net_branch: list[int],
        widths_net_trunk: list[int],
        n_embed_h: int,
        n_intermediate: int,
        activation_deepset,
        activation_branch,
        activation_trunk,
        final_activation_deepset,
        final_activation_branch,
        final_activation_trunk,
        key,
    ):
        ks = jr.split(key, 10)
        n_xs = data.cond_context.shape[-1]
        trunk_dim = data.xs_query.shape[-1]

        net_deepset = MLP(
            n_xs * 2,
            n_embed_h,
            widths_net_deepset,
            activation_deepset,
            final_activation_deepset,
            key=ks[0],
        )
        net_branch = MLP(
            n_embed_h + n_xs,
            n_intermediate,
            widths_net_branch,
            activation_branch,
            final_activation_branch,
            key=ks[1],
        )
        net_trunk = MLP(
            trunk_dim,
            n_intermediate,
            widths_net_trunk,
            activation_trunk,
            final_activation_trunk,
            key=ks[2],
        )
        return cls(DeepSet(net_deepset, pooling=pooling), DeepONet(net_branch, net_trunk, key=ks[3]))

    @eqx.filter_jit
    def __call__(self, cond_context, qoi_context, cond_query, xs_query):
        context_pairs = jnp.concatenate([cond_context, qoi_context], axis=-1)
        embedding = self.setnet(context_pairs)
        branch_input = jnp.concatenate([embedding, cond_query], axis=-1)
        fixed_operator = self.deeponet.get_net_with_fixed_condition(branch_input)
        return jax.vmap(fixed_operator)(xs_query)

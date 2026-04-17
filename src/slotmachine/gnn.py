from collections.abc import Callable
from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array


class GNN(eqx.Module):
    node_size: int
    edge_size: int
    context_size: int
    node_depth: int
    edge_depth: int
    context_depth: int
    aggregation: Callable

    node_net: eqx.nn.MLP
    edge_net: eqx.nn.MLP
    context_net: eqx.nn.MLP | None

    def __init__(
        self,
        node_size: int,
        edge_size: int,
        context_size: int,
        node_depth: int,
        edge_depth: int,
        context_depth: int,
        aggregation: str,
        *,
        slot_size: int,
        action_size: int,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 3)

        self.node_size = node_size
        self.edge_size = edge_size
        self.node_depth = node_depth
        self.edge_depth = edge_depth
        self.context_depth = context_depth

        if action_size > 0:
            self.context_net = eqx.nn.MLP(
                action_size, slot_size, context_size, context_depth, key=keys[2]
            )
        else:
            self.context_net = None
            context_size = 0

        self.context_size = context_size

        self.node_net = eqx.nn.MLP(
            2 * slot_size + context_size, slot_size, node_size, node_depth, key=keys[0]
        )
        self.edge_net = eqx.nn.MLP(
            2 * slot_size + context_size, slot_size, edge_size, edge_depth, key=keys[1]
        )

        match aggregation:
            case "max":
                self.aggregation = partial(jnp.max, axis=0)
            case "mean":
                self.aggregation = partial(jnp.mean, axis=0)
            case _:
                raise f"Aggregation type {aggregation} not implemented"

    def __call__(
        self,
        input: Float[Array, "num_slots slot_size"],
        action: Float[Array, "action_size"] | None = None,
    ) -> Float[Array, "num_slots slot_size"]:
        num_slots = input.shape[0]
        slot_size = input.shape[1]

        # add context
        if not action is None:
            context = self.context_net(action)[None, :]  # [1 slot_size]
            context = jnp.repeat(context, num_slots, axis=0)  # [num_slots slot_size]
            augmented = jnp.concatenate(
                (input, context), axis=1
            )  # [num_slots 2*slot_size]
        else:
            augmented = input

        # create edge pairs
        idx = jnp.arange(num_slots - 1)  # [num_slots-1]
        shift_mask = jnp.triu(
            jnp.ones((num_slots, num_slots - 1), dtype=jnp.int32)
        )  # [num_slots num_slots-1]
        shift_idx = idx[None, :] + shift_mask  # [num_slots num_slots-1]
        a = jnp.repeat(
            input[:, None, :], num_slots - 1, axis=1
        )  # [num_slots num_slots-1 slot_size]
        b = augmented[shift_idx]  # [num_slots num_slots-1 slot_size+context_size]
        pairs = jnp.concatenate(
            (a, b), axis=2
        )  # [num_slots num_slots-1 2*slot_size+context_size]
        flat_pairs = jnp.reshape(pairs, (-1, 2 * slot_size + self.context_size))

        # evaluate and aggregate edges
        edge = eqx.filter_vmap(self.edge_net)(
            flat_pairs
        )  # [num_slots*(num_slots-1) slot_size]
        edge = jnp.reshape(edge, (num_slots, num_slots - 1, slot_size))
        aggregated = eqx.filter_vmap(self.aggregation)(edge)  # [num_slots slot_size]

        # evaluate nodes
        full = jnp.concatenate(
            (augmented, aggregated), axis=1
        )  # [num_slots 2*slot_size+context_size]
        node = eqx.filter_vmap(self.node_net)(full)  # [num_slots slot_size]

        return node

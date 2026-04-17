import jax.numpy as jnp
from jaxtyping import Int, Float, Array

FILTER_PENALTY = 5000


def match(
    z_o: Float[Array, "num_slots z_dim"],
    z_p: Float[Array, "num_slots z_dim"],
    magnitude: Float[Array, "1"],
    property: Int[Array, "1"],
    match_direct: bool = False,
    bg_idx_o: Int[Array, "1"] | None = None,
    bg_idx_p: Int[Array, "1"] | None = None,
):
    num_slots = z_o.shape[0]

    idx = jnp.full((num_slots, 1), property)

    penalty_o = jnp.zeros((num_slots,))
    penalty_p = jnp.zeros((num_slots,))
    if not bg_idx_o is None:
        penalty_o = penalty_o.at[bg_idx_o].set(FILTER_PENALTY)
    if not bg_idx_p is None:
        penalty_p = penalty_p.at[bg_idx_p].set(FILTER_PENALTY)

    z_o = jnp.take_along_axis(z_o, idx, axis=1)  # [num_slots 1]
    z_p = jnp.take_along_axis(z_p, idx, axis=1)  # [num_slots 1]

    if not match_direct:
        z_o = jnp.repeat(
            z_o, num_slots, axis=0
        )  # [num_slots*num_slots 1] [a a b b c c]
        z_p = jnp.tile(z_p, (num_slots, 1))  # [num_slots*num_slots 1] [a b c a b c]

        penalty_o = jnp.repeat(penalty_o, num_slots)
        penalty_p = jnp.tile(penalty_p, (penalty_p,))

    diff = jnp.square(z_p - z_o - magnitude) + penalty_o + penalty_p
    return jnp.min(diff)


def identify_background(
    attn: Float[Array, "num_slots num_inputs"],
) -> Int:
    std = jnp.std(attn, axis=1)
    return jnp.argmin(std)

import jax
from jax import numpy as jnp
import equinox as eqx
from equinox import nn
from jaxtyping import Array, Float
from functools import partial


def sinkhorn(
    c: Float[Array, "num_inputs num_slots"],
    a: Float[Array, "num_inputs"],
    b: Float[Array, "num_slots"],
    u: Float[Array, "num_inputs"],
    v: Float[Array, "num_slots"],
    n_sh_iters: int = 5,
):
    p = -c
    log_a = jnp.log(a)
    log_b = jnp.log(b)

    def f(carry, _, p, log_a, log_b):
        new_u = log_a - jax.scipy.special.logsumexp(
            p + jnp.expand_dims(carry["v"], 0), axis=1
        )  # [num_inputs]
        new_v = log_b - jax.scipy.special.logsumexp(
            p + jnp.expand_dims(new_u, 1), axis=0
        )  # [num_slots]
        new_carry = dict(u=new_u, v=new_v)

        return new_carry, None

    out, _ = jax.lax.scan(
        partial(f, p=p, log_a=log_a, log_b=log_b), dict(u=u, v=v), length=n_sh_iters
    )

    logT = (
        p + jnp.expand_dims(out["u"], 1) + jnp.expand_dims(out["v"], 0)
    )  # [num_inputs num_slots]
    attn = jnp.exp(logT)

    return attn, out["u"], out["v"]


def sinkhorn_entropy(
    c: Float[Array, "num_inputs num_slots"],
    a: Float[Array, "num_inputs"],
    b: Float[Array, "num_slots"],
    u: Float[Array, "num_inputs"],
    v: Float[Array, "num_slots"],
    n_sh_iters: int = 5,
):
    attn, u, v = sinkhorn(c, a, b, u, v, n_sh_iters=n_sh_iters)
    entropy = jnp.mean(
        jax.scipy.special.entr(jnp.clip(attn, min=1e-20, max=1))
    )  # scalar
    return entropy, dict(u=u, v=v)


def minimize_sinkhorn_entropy(
    c0: Float[Array, "num_inputs num_slots"],
    a: Float[Array, "num_inputs"],
    b: Float[Array, "num_slots"],
    mesh_lr: int = 1,
    n_mesh_iters: int = 4,
    n_sh_iters: int = 5,
    *,
    key: jax.random.PRNGKey,
):
    noise = jax.random.normal(key, c0.shape)

    ct = c0 + 0.001 * noise

    def f(carry, _, a, b):
        grad, aux = jax.grad(sinkhorn_entropy, has_aux=True)(
            carry["ct"], a, b, carry["u"], carry["v"], n_sh_iters=n_sh_iters
        )  # [num_inputs num_slots]
        grad = jnp.linalg.norm(grad + 1e-20, axis=(0, 1))  # scalar
        ct = carry["ct"] - mesh_lr * grad

        return dict(ct=ct, u=aux["u"], v=aux["v"]), None

    out, _ = jax.lax.scan(
        partial(f, a=a, b=b),
        dict(ct=ct, u=jnp.zeros(a.shape), v=jnp.zeros(b.shape)),
        length=n_mesh_iters,
    )

    return out["ct"], out["u"], out["v"]


def slot_update(
    carry: dict,
    _,
    self: eqx.Module,
    a: Float[Array, "num_inputs"],
    k: Float[Array, "num_inputs slot_size"],
    v: Float[Array, "num_inputs slot_size"],
    num_slots: int,
):
    keys = jax.random.split(carry["key"], 2)
    slots_prev = carry["slots"]  # [num_slots slot_size]
    slots = jax.vmap(self.slot_norm)(carry["slots"])

    b = jax.vmap(self.slot_weight)(slots)  # [num_slots 1]
    b = jax.nn.softmax(jnp.squeeze(b)) * num_slots  # [num_slots]

    q = jax.vmap(self.query_map)(slots)  # [num_slots slot_size]

    attn_logits = jax.vmap(
        lambda x, y: jnp.linalg.norm(x - y, axis=1), in_axes=(0, None)
    )(
        k, q
    )  # [num_inputs num_slots]

    attn_logits, p, q = minimize_sinkhorn_entropy(
        attn_logits, a, b, mesh_lr=5, key=keys[0]
    )
    attn, _, _ = sinkhorn(attn_logits, a, b, p, q)  # [num_inputs num_slots]

    updates = jnp.matmul(jnp.transpose(attn), v)  # [num_slots slot_size]

    slots = jax.vmap(self.gru, in_axes=(0, 0))(
        updates, slots_prev
    )  # [num_slots slot_size]
    normed = jax.vmap(self.mlp_norm)(slots)
    slots += jax.vmap(self.mlp)(normed)

    return dict(slots=slots, attn=attn, key=keys[1]), None


class SlotAttention(eqx.Module):
    feature_size: int
    slot_size: int
    mlp_size: int
    epsilon: float

    input_norm: nn.LayerNorm
    slot_norm: nn.LayerNorm
    mlp_norm: nn.LayerNorm

    query_map: nn.Linear
    key_map: nn.Linear
    value_map: nn.Linear

    input_weight: nn.Linear
    slot_weight: nn.Linear

    gru: nn.GRUCell
    mlp: nn.Sequential

    mu: Array
    log_sigma: Array

    def __init__(
        self,
        feature_size: int,
        slot_size: int,
        mlp_size: int,
        epsilon: float,
        *,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 8)

        self.feature_size = feature_size
        self.slot_size = slot_size
        self.mlp_size = mlp_size
        self.epsilon = epsilon

        self.input_norm = nn.LayerNorm(feature_size)
        self.slot_norm = nn.LayerNorm(slot_size)
        self.mlp_norm = nn.LayerNorm(slot_size)

        self.query_map = nn.Linear(slot_size, slot_size, use_bias=False, key=keys[0])
        self.key_map = nn.Linear(feature_size, slot_size, use_bias=False, key=keys[1])
        self.value_map = nn.Linear(feature_size, slot_size, use_bias=False, key=keys[2])

        self.input_weight = nn.Linear(feature_size, 1, key=keys[3])
        self.slot_weight = nn.Linear(slot_size, 1, key=keys[4])

        self.gru = nn.GRUCell(slot_size, slot_size, key=keys[5])
        self.mlp = nn.Sequential(
            [
                nn.Linear(slot_size, mlp_size, key=keys[6]),
                eqx.nn.Lambda(jax.nn.relu),
                nn.Linear(mlp_size, slot_size, key=keys[7]),
            ]
        )

        initializer = jax.nn.initializers.glorot_uniform()

        self.mu = initializer(
            keys[8],
            (
                1,
                slot_size,
            ),
        )
        self.log_sigma = initializer(
            keys[9],
            (
                1,
                slot_size,
            ),
        )

    def __call__(
        self,
        inputs: Float[Array, "num_inputs feature_size"],
        num_slots: int,
        num_iterations: int,
        key: jax.random.PRNGKey,
        slots_init: Float[Array, "{num_slots} slot_size"] | None = None,
    ):
        keys = jax.random.split(key, 3)
        num_inputs = inputs.shape[0]

        inputs = jax.vmap(self.input_norm)(inputs)

        k = jax.vmap(self.key_map)(inputs)  # [num_inputs slot_size]
        v = jax.vmap(self.value_map)(inputs)  # [num_inputs slot_size]

        if slots_init is None:
            slots_init = jax.random.normal(keys[0], (num_slots, self.slot_size))
            slots = (
                jax.lax.stop_gradient(self.mu)
                + jax.lax.stop_gradient(jnp.exp(self.log_sigma)) * slots_init
            )  # [num_slots slot_size]
        else:
            slots = slots_init

        a = jax.vmap(self.input_weight)(inputs)  # [num_inputs 1]
        a = jax.nn.softmax(jnp.squeeze(a)) * num_slots  # [num_inputs]

        # iterate n-1 times, then truncate gradient
        out, _ = jax.lax.scan(
            partial(slot_update, self=self, a=a, k=k, v=v, num_slots=num_slots),
            dict(slots=slots, attn=jnp.zeros((num_inputs, num_slots)), key=keys[1]),
            length=num_iterations - 1,
        )
        slots = jax.lax.stop_gradient(out["slots"])
        # final iteration
        out, _ = slot_update(
            dict(slots=slots, attn=jnp.zeros((num_inputs, num_slots)), key=keys[2]),
            None,
            self,
            a,
            k,
            v,
            num_slots,
        )

        attn = jnp.transpose(out["attn"])
        return out["slots"], attn

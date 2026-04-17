import jax
from jax import numpy as jnp
import equinox as eqx
from equinox import nn
from jaxtyping import Array, Float


def build_grid(
    width: int, height: int, depth: int
) -> Float[Array, "{width} {height} {depth} 4"]:
    horizontal = jnp.linspace(0.0, 1.0, width)
    vertical = jnp.linspace(0.0, 1.0, height)
    longitudinal = jnp.linspace(0.0, 1.0, depth)
    grid = jnp.meshgrid(*[horizontal, vertical, longitudinal], indexing="ij")
    grid = jnp.stack(grid, axis=-1)
    grid = jnp.reshape(grid, (width, height, depth, -1))
    grid = jnp.concatenate((grid, 1.0 - grid), axis=-1)
    return grid


class PositionalEmbedding(eqx.Module):
    hidden_size: int
    width: int
    height: int
    depth: int

    linear: nn.Linear
    grid: Array

    def __init__(
        self,
        width: int,
        height: int,
        depth: int,
        hidden_size: int,
        *,
        key: jax.random.PRNGKey,
    ):
        self.width = width
        self.height = height
        self.depth = depth
        self.hidden_size = hidden_size

        self.linear = nn.Linear(6, hidden_size, key=key)
        self.grid = build_grid(width, height, depth)

    def __call__(self, input: Float[Array, "width*height*depth hidden_size"]):
        y = jax.vmap(self.linear)(
            jax.lax.stop_gradient(jnp.reshape(self.grid, (-1, 6)))
        )  # [width*height*depth hidden_size]
        return input + y


class Encoder(eqx.Module):
    width: int
    height: int
    feature_size: int
    seq_length: int

    cnn: nn.Sequential
    embedding: PositionalEmbedding
    norm: nn.LayerNorm
    mlp: nn.Sequential

    def __init__(
        self,
        width: int,
        height: int,
        feature_size: int,
        seq_length: int,
        *,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 7)

        self.width = width
        self.height = height
        self.feature_size = feature_size
        self.seq_length = seq_length

        self.cnn = nn.Sequential(
            [
                nn.Conv2d(3, feature_size, 5, padding="SAME", key=keys[0]),
                nn.Lambda(jax.nn.relu),
                nn.Conv2d(feature_size, feature_size, 5, padding="SAME", key=keys[1]),
                nn.Lambda(jax.nn.relu),
                nn.Conv2d(feature_size, feature_size, 5, padding="SAME", key=keys[2]),
                nn.Lambda(jax.nn.relu),
                nn.Conv2d(feature_size, feature_size, 5, padding="SAME", key=keys[3]),
                nn.Lambda(jax.nn.relu),
            ]
        )

        self.embedding = PositionalEmbedding(
            width, height, seq_length, feature_size, key=keys[4]
        )

        self.norm = nn.LayerNorm(feature_size)

        self.mlp = nn.Sequential(
            [
                nn.Linear(feature_size, feature_size, key=keys[5]),
                nn.Lambda(jax.nn.relu),
                nn.Linear(feature_size, feature_size, key=keys[6]),
            ]
        )

    def __call__(self, input: Float[Array, "seq_length 3 width height"]):
        x = jax.vmap(self.cnn)(input)  # [seq_length feature_size width height]
        x = jnp.transpose(x, (2, 3, 0, 1))  # [width height seq_length feature_size]
        x = jnp.reshape(
            x, (-1, self.feature_size)
        )  # [width*height*seq_length feature_size]
        x = self.embedding(x)  # [width*height*seq_length feature_size]
        x = jax.vmap(self.norm)(x)
        x = jax.vmap(self.mlp)(x)  # [width*height*seq_length feature_size]
        return x


class Decoder(eqx.Module):
    width: int
    height: int
    slot_size: int
    seq_length: int

    init_width: int
    init_height: int

    embedding: PositionalEmbedding
    cnn: nn.Sequential

    def __init__(
        self,
        width: int,
        height: int,
        slot_size: int,
        seq_length: int,
        *,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 7)

        self.width = width
        self.height = height
        self.slot_size = slot_size
        self.seq_length = seq_length

        self.init_width = width // 16
        self.init_height = height // 16

        self.embedding = PositionalEmbedding(
            self.init_width, self.init_height, self.seq_length, slot_size, key=keys[0]
        )

        cnn_params = dict(
            in_channels=slot_size,
            out_channels=slot_size,
            kernel_size=5,
            padding="SAME",
            stride=2,
        )
        self.cnn = nn.Sequential(
            [
                nn.ConvTranspose2d(
                    **cnn_params,
                    key=keys[1],
                ),
                nn.Lambda(jax.nn.relu),
                nn.ConvTranspose2d(
                    **cnn_params,
                    key=keys[2],
                ),
                nn.Lambda(jax.nn.relu),
                nn.ConvTranspose2d(
                    **cnn_params,
                    key=keys[3],
                ),
                nn.Lambda(jax.nn.relu),
                nn.ConvTranspose2d(
                    **cnn_params,
                    key=keys[4],
                ),
                nn.Lambda(jax.nn.relu),
                nn.ConvTranspose2d(
                    slot_size,
                    slot_size,
                    5,
                    stride=1,
                    padding="SAME",
                    key=keys[5],
                ),
                nn.Lambda(jax.nn.relu),
                nn.ConvTranspose2d(
                    slot_size,
                    4,
                    3,
                    stride=1,
                    padding="SAME",
                    key=keys[6],
                ),
            ]
        )

    def __call__(self, input: Float[Array, "num_slots slot_size"]):
        x = jnp.expand_dims(input, (1,))  # [num_slots 1 slot_size]
        x = jnp.tile(
            x, (1, self.init_width * self.init_height * self.seq_length, 1)
        )  # [num_slots init_width*init_height*seq_length slot_size]
        x = jax.vmap(self.embedding)(x)
        x = jnp.reshape(
            x, (-1, self.init_width, self.init_height, self.seq_length, self.slot_size)
        )
        x = jnp.transpose(
            x, (3, 0, 4, 1, 2)
        )  # [seq_length num_slots slot_size init_width init_height]
        x = jax.vmap(jax.vmap(self.cnn))(x)  # [seq_length num_slots 4 width height]
        recon, masks = jnp.split(x, (3,), axis=2)
        masks = jax.nn.softmax(masks, axis=1)
        recon_combined = jnp.sum(recon * masks, axis=1)  # [seq_length 3 width height]

        return recon_combined, recon, masks


class ActionEncoder(eqx.Module):
    action_size: int
    mlp_size: int
    feature_size: int
    seq_length: int

    mlp: nn.Sequential
    embedding: nn.Linear
    offsets: Array

    def __init__(
        self,
        action_size: int,
        mlp_size: int,
        feature_size: int,
        seq_length: int,
        *,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 5)

        self.action_size = action_size
        self.mlp_size = mlp_size
        self.feature_size = feature_size
        self.seq_length = seq_length

        self.mlp = nn.Sequential(
            [
                nn.Linear(action_size, mlp_size, key=keys[0]),
                eqx.nn.Lambda(jax.nn.relu),
                nn.Linear(mlp_size, mlp_size, key=keys[1]),
                eqx.nn.Lambda(jax.nn.relu),
                nn.Linear(mlp_size, mlp_size, key=keys[2]),
                eqx.nn.Lambda(jax.nn.relu),
                nn.Linear(mlp_size, feature_size, key=keys[3]),
            ]
        )

        self.embedding = nn.Linear(2, feature_size, key=keys[4])

        seq = jnp.linspace(0.0, 1.0, seq_length - 1)  # [seq_length-1]
        seq = jnp.expand_dims(seq, axis=1)  # [seq_length-1 1]
        self.offsets = jnp.concatenate((seq, 1 - seq), axis=1)  # [seq_length-1 2]

    def __call__(self, actions: Float[Array, "seq_length-1 action_size"]):
        x = jax.vmap(self.mlp)(actions)  # [seq_length-1 feature_size]
        y = jax.vmap(self.embedding)(
            jax.lax.stop_gradient(self.offset)
        )  # [seq_length-1 feature_size]
        return x + y

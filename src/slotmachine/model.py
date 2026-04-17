import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array

from .autoencoder import Encoder, Decoder, ActionEncoder
from .slotattention import SlotAttention
from .disentanglement import make_ensemble


class Slotmachine(eqx.Module):
    obs_width: int
    obs_height: int
    feature_size: int
    slot_size: int
    mlp_size: int
    latent_size: int
    seq_length: int
    action_size: int

    encoder: Encoder
    decoder: Decoder
    slot_attention: SlotAttention
    disentanglement: eqx.Module
    action_encoder: ActionEncoder

    def __init__(
        self,
        obs_width: int,
        obs_height: int,
        feature_size: int,
        slot_size: int,
        mlp_size: int,
        latent_size: int,
        seq_length: int,
        action_size: int,
        dynamics: bool,
        *,
        key: jax.random.PRNGKey,
    ):
        keys = jax.random.split(key, 5)

        self.obs_width = obs_width
        self.obs_height = obs_height
        self.feature_size = feature_size
        self.slot_size = slot_size
        self.mlp_size = mlp_size
        self.latent_size = latent_size
        self.seq_length = seq_length
        self.action_size = action_size

        self.encoder = Encoder(
            obs_width, obs_height, feature_size, seq_length, key=keys[0]
        )
        self.decoder = Decoder(
            obs_width, obs_height, slot_size, seq_length, key=keys[1]
        )
        self.slot_attention = SlotAttention(
            feature_size, slot_size, mlp_size, 1e-8, key=keys[2]
        )
        if action_size == 0 or seq_length == 1:
            self.action_encoder = None
        else:
            self.action_encoder = ActionEncoder(
                action_size, mlp_size, feature_size, seq_length, key=keys[3]
            )

        keys = jax.random.split(keys[4], latent_size)
        self.disentanglement = make_ensemble(keys, slot_size)

    def __call__(
        self,
        input: Float[Array, "seq_length 3 width height"],
        num_slots: int,
        num_iterations: int,
        key: jax.random.PRNGKey,
        slots_init: Float[Array, "{num_slots} slot_size"] | None = None,
        actions: Float[Array, "seq_length-1 action_size"] | None = None,
    ):
        enc = self.encoder(input)
        if not actions is None:
            act_enc = self.action_encoder(actions)
            enc = jnp.concatenate((enc, act_enc), axis=0)
        slots, attn = self.slot_attention(
            enc, num_slots, num_iterations, key, slots_init
        )
        recon_combined, recon, masks = self.decoder(slots)
        z = eqx.filter_vmap(self.disentanglement)(slots)  # [num_slots latent_size 1]
        z = jnp.squeeze(z)  # [num_slots latent_size]
        return slots, z, attn, recon_combined, recon, masks

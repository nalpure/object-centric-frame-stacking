import jax
import equinox as eqx

import slotmachine.utils as utl
from slotmachine.model import Slotmachine

config = utl.load_config(f"cfgs/seq_sa.toml")
key = jax.random.PRNGKey(666)

keys = jax.random.split(key, 4)
model = Slotmachine(key=keys[0], **config["model"])

x = jax.random.uniform(keys[1], (4, 3, 64, 64))
actions = jax.random.uniform(keys[2], (3, 16))

model(x, 4, 3, key=keys[3])

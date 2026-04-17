import jax
import equinox as eqx
from equinox import nn
from jaxtyping import Array, Float
from sklearn import metrics
from sklearn import linear_model


@eqx.filter_vmap
def make_ensemble(key: jax.random.PRNGKey, slot_size: int):
    return DisentanglementHead(slot_size, key=key)


@eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
def evaluate_ensemble(model, x):
    return model(x)


class DisentanglementHead(eqx.Module):
    slot_size: int

    mlp: nn.Sequential

    def __init__(self, slot_size: int, *, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 4)

        self.slot_size = slot_size

        self.mlp = nn.Sequential(
            [
                nn.Linear(slot_size, slot_size // 2, key=keys[0]),
                nn.Lambda(jax.nn.relu),
                nn.Linear(slot_size // 2, slot_size // 2, key=keys[1]),
                nn.Lambda(jax.nn.relu),
                nn.Linear(slot_size // 2, slot_size // 4, use_bias=False, key=keys[2]),
                nn.Lambda(jax.nn.relu),
                nn.Linear(slot_size // 4, 1, use_bias=False, key=keys[4]),
            ]
        )

    def __call__(self, input: Float[Array, "slot_size"]):
        return evaluate_ensemble(self.mlp, input)


def mcc(
    z_true: Float[Array, "num_objects latent_size"],
    z_pred: Float[Array, "num_objects latent_size"],
):
    model = linear_model.LinearRegression()
    model.fit(z_pred, z_true)
    z_model = model.predict(z_pred)
    score = metrics.r2_score(z_true, z_model)
    return score

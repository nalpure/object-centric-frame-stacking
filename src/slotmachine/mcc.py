import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
import optax
from sklearn import metrics, linear_model


def get_mcc(truth: Float[Array, "data_size num_slots latent_size"], pred: Float[Array, "data_size num_slots latent_size"]):
    latent_size = truth.shape[2]

    flat_truth = jnp.reshape(truth, (-1, latent_size))
    flat_pred = jnp.reshape(pred, (-1, latent_size))

    center_truth = flat_truth - jnp.mean(flat_truth, axis=0)
    center_pred = flat_pred - jnp.mean(flat_pred, axis=0)

    num = jnp.sum(center_truth * center_pred, axis=0)
    den = jnp.sqrt(jnp.sum(jnp.square(center_truth), axis=0) * jnp.sum(jnp.square(center_pred), axis=0))

    pearson = num / den # [latent_size]
    mcc = jnp.mean(pearson)

    return mcc, pearson

def get_linear_disentanglement(
    truth: Float[Array, "data_size num_slots latent_size"],
    pred: Float[Array, "data_size num_slots prediction_size"],
):
    num_slots = truth.shape[1]
    latent_size = truth.shape[2]
    pred_size = pred.shape[2]
    flat_truth = jnp.reshape(truth, (-1, latent_size))
    flat_pred = jnp.reshape(pred, (-1, pred_size))

    # perform initial fit
    model = linear_model.LinearRegression()
    model.fit(flat_pred, flat_truth)
    regressed = model.predict(flat_pred)
    old_score = metrics.r2_score(flat_truth, regressed)
    print(f"Initial score: {old_score}")

    def f(prediction, regression, target):
        cost_matrix = jnp.sum(
            jnp.square(target[:, None, :] - regression[None, :, :]), axis=2
        )
        row, col = optax.assignment.hungarian_algorithm(cost_matrix)
        row_inv = jnp.argsort(row)
        perm = col[row_inv]

        return prediction[perm]

    # iteratively rematch
    rematch = jax.jit(jax.vmap(f))
    converged = False
    while not converged:
        regressed = jnp.reshape(regressed, (-1, num_slots, latent_size))

        pred = rematch(pred, regressed, truth)
        flat_pred = jnp.reshape(pred, (-1, pred_size))

        model = linear_model.LinearRegression()
        model.fit(flat_pred, flat_truth)
        regressed = model.predict(flat_pred)
        score = metrics.r2_score(flat_truth, regressed)
        print(f"New score: {score}")

        if score <= old_score:
            converged = True
        else:
            old_score = score

    return old_score


def find_alignment(attn, masks_pred, masks):
    seq_length = masks_pred.shape[0]
    num_slots = attn.shape[0]

    # drop background slots
    drop_idx = jnp.argsort(jnp.std(attn, axis=1))  # [num_slots]

    masks_pred = jnp.squeeze(masks_pred)  # [seq_length num_slots width height]
    masks_pred = masks_pred[:, drop_idx, :, :][
        :, 1:
    ]  # [seq_length num_slots-1 width height]

    # solve assignment problem
    flat_truth = jnp.reshape(masks, (seq_length, num_slots - 1, -1))
    flat_pred = jnp.reshape(masks_pred, (seq_length, num_slots - 1, -1))

    cost_matrix = jnp.sum(
        jnp.square(flat_truth[:, :, None, :] - flat_pred[:, None, :, :]), axis=(0, 3)
    )
    row, col = optax.assignment.hungarian_algorithm(cost_matrix)

    row_inv = jnp.argsort(row)
    perm = col[row_inv]

    return drop_idx, perm


def apply_alignment(x, drop_idx, perm):
    x = x[drop_idx][1:]
    x = x[perm]
    return x

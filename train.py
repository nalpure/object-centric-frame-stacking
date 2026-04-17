import argparse
import numpy as np
import os
import jax
import jax.sharding as jshard
import jax.numpy as jnp
import optax
import equinox as eqx
from tqdm import tqdm
import jax_dataloader as jdl
import time

from slotmachine.model import Slotmachine
import slotmachine.utils as utl
from slotmachine.mcc import find_alignment, apply_alignment, get_linear_disentanglement, get_mcc
from slotmachine.match import match, identify_background


def preprocess_batch(img_o, img_p, actions, config):
    seq_length = config["model"]["seq_length"]

    in_o = img_o[:, :seq_length]
    in_o /= 255

    in_p = img_p[:, :seq_length]
    in_p /= 255

    if config["model"]["dynamics"]:
        out_o = img_o[:, 1 : seq_length + 1]
        out_o /= 255
    else:
        out_o = in_o

    if config["model"]["action_size"] == 0:
        actions = None
    else:
        actions = actions

    return in_o, in_p, out_o, actions


def select_match_type(slots, type):
    if type == "direct":
        slots_in = slots
        match_direct = True
    elif type == "full":
        slots_in = None
        match_direct = False
    else:
        raise (f"Slot matching procedure {type} not implemented!")

    return slots_in, match_direct


def select_bg_filtering(attn_o, attn_p, filter):
    if filter:
        bg_idx_o = eqx.filter_vmap(identify_background)(attn_o)
        bg_idx_p = eqx.filter_vmap(identify_background)(attn_p)
    else:
        bg_idx_o = None
        bg_idx_p = None

    return bg_idx_o, bg_idx_p


def eval_step(model, batch, config, key):
    key, subkey = jax.random.split(key, 2)
    info = {}

    img_o, img_p, magnitudes, properties, actions, masks, _ = batch
    in_o, in_p, out_o, actions = preprocess_batch(img_o, img_p, actions, config)
    masks = masks[:, : config["model"]["seq_length"]]

    batch_size = in_o.shape[0]
    num_slots = config["slot"]["num_slots"]
    sa_iter = config["slot"]["sa_iterations"]

    key, subkey = jax.random.split(key, 2)
    keys = jax.random.split(subkey, batch_size)
    slots, z_o, attn_o, recon_combined, _, masks_pred = eqx.filter_vmap(model)(
        in_o, num_slots, sa_iter, keys, None, actions
    )

    if config["eval"]["recon_loss"]:
        info["reconstruction_loss"] = jnp.mean(jnp.square(out_o - recon_combined))

    if config["eval"]["dis_loss"]:
        slots_in, match_direct = select_match_type(slots, config["eval"]["match_type"])
        keys = jax.random.split(key, batch_size)
        _, z_p, attn_p, _, _, _ = eqx.filter_vmap(model)(
            in_p, num_slots, sa_iter, keys, slots_in, actions
        )

        bg_idx_o, bg_idx_p = select_bg_filtering(
            attn_o, attn_p, config["eval"]["filter_background"]
        )
        min_diff = eqx.filter_vmap(match)(
            z_o, z_p, magnitudes, properties, match_direct, bg_idx_o, bg_idx_p
        )
        info["disentanglement_loss"] = jnp.mean(min_diff)

    if config["eval"]["latent_ld"] or config["eval"]["slot_ld"]:
        drop_idx, perm = eqx.filter_vmap(find_alignment)(attn_o, masks_pred, masks)

    if config["eval"]["latent_ld"]:
        z_align = eqx.filter_vmap(apply_alignment)(z_o, drop_idx, perm)
    else:
        z_align = None

    if config["eval"]["slot_ld"]:
        slots_align = eqx.filter_vmap(apply_alignment)(slots, drop_idx, perm)
    else:
        slots_align = None

    return info, z_align, slots_align


def train_loss(model, batch, config, key):
    key, subkey = jax.random.split(key, 2)
    info = {}

    img_o, img_p, magnitudes, properties, actions = batch
    in_o, in_p, out_o, actions = preprocess_batch(img_o, img_p, actions, config)

    batch_size = in_o.shape[0]
    num_slots = config["slot"]["num_slots"]
    sa_iter = config["slot"]["sa_iterations"]
    loss = 0

    keys = jax.random.split(subkey, batch_size)
    slots, z_o, attn_o, recon_combined, _, _ = eqx.filter_vmap(model)(
        in_o, num_slots, sa_iter, keys, None, actions
    )

    if config["train"]["recon_loss"]:
        info["reconstruction_loss"] = jnp.mean(jnp.square(out_o - recon_combined))
        loss += config["train"]["recon_w"] * info["reconstruction_loss"]

    if config["train"]["dis_loss"]:
        slots_in, match_direct = select_match_type(slots, config["train"]["match_type"])
        keys = jax.random.split(key, batch_size)
        _, z_p, attn_p, _, _, _ = eqx.filter_vmap(model)(
            in_p, num_slots, sa_iter, keys, slots_in, actions
        )

        bg_idx_o, bg_idx_p = select_bg_filtering(
            attn_o, attn_p, config["train"]["filter_background"]
        )
        min_diff = eqx.filter_vmap(match)(
            z_o, z_p, magnitudes, properties, match_direct, bg_idx_o, bg_idx_p
        )
        info["disentanglement_loss"] = jnp.mean(min_diff)
        loss += config["train"]["dis_w"] * info["disentanglement_loss"]

    return loss, info


@eqx.filter_jit(donate="all")
def train_step(model, batch, tx, opt_state, config, key, sharding, replicated):
    model, opt_state = eqx.filter_shard((model, opt_state), replicated)
    batch = eqx.filter_shard(batch, sharding)

    grads, info = eqx.filter_grad(train_loss, has_aux=True)(model, batch, config, key)
    updates, opt_state = tx.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    model, opt_state = eqx.filter_shard((model, opt_state), replicated)
    return info, model, opt_state


def train(model, dataloader, run_name, config, key):
    metrics = []
    start_time = time.time()
    tx = optax.adamw(
        learning_rate=config["train"]["opt"]["lr"],
        weight_decay=config["train"]["opt"]["wd"],
    )
    if "clip" in config["train"]["opt"]:
        tx = optax.chain(optax.clip_by_global_norm(config["train"]["opt"]["clip"]), tx)
    opt_state = tx.init(eqx.filter(model, eqx.is_inexact_array))
    batches_per_epoch = len(dataloader)

    num_devices = len(jax.devices())
    mesh = jax.make_mesh((num_devices,), ("batch",))
    sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
    replicated = jshard.NamedSharding(mesh, jshard.PartitionSpec())

    model = eqx.filter_shard(model, replicated)
    for epoch in tqdm(range(config["train"]["epochs"])):
        epoch_metric = {}
        for batch in dataloader:
            batch = eqx.filter_shard(batch, sharding)
            key, subkey = jax.random.split(key, 2)
            info, model, opt_state = train_step(
                model, batch, tx, opt_state, config, subkey, sharding, replicated
            )
            utl.aggregate_metrics(epoch_metric, info)

        utl.normalize_metrics(epoch_metric, batches_per_epoch)
        current_time = time.time()
        epoch_metric["time"] = (current_time - start_time) / 60
        metrics.append(epoch_metric)

        if (epoch + 1) % config["train"]["log_rate"] == 0 or epoch + 1 == config[
            "train"
        ]["epochs"]:
            utl.save_metrics(metrics, run_name, "train_metrics.csv")
            metrics.clear()

        if (
            config["train"]["ckpt_rate"] > 0
            and (epoch + 1) % config["train"]["ckpt_rate"] == 0
            or epoch + 1 == config["train"]["epochs"]
        ):
            eqx.tree_serialise_leaves(f"out/{run_name}/ckpt_{epoch+1}.eqx", model)


def eval(model, dataloader, run_name, config, key):
    metrics = {}
    num_devices = len(jax.devices())
    mesh = jax.make_mesh((num_devices,), ("batch",))
    sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
    replicated = jshard.NamedSharding(mesh, jshard.PartitionSpec())
    batches_per_epoch = len(dataloader)

    truth = []
    z = []
    slots = []

    model = eqx.filter_shard(model, replicated)
    for batch in tqdm(dataloader):
        _, _, _, _, _, _, truth_batch = batch
        batch = eqx.filter_shard(batch, sharding)
        key, subkey = jax.random.split(key, 2)

        batch_metrics, z_batch, slots_batch = eval_step(model, batch, config, subkey)

        utl.aggregate_metrics(metrics, batch_metrics)
        truth.append(jnp.squeeze(truth_batch[:, 0]))
        z.append(z_batch)
        slots.append(slots_batch)

    utl.normalize_metrics(metrics, batches_per_epoch)

    truth = jnp.concatenate(truth, axis=0)
    if config["eval"]["latent_ld"]:
        x = jnp.concatenate(z, axis=0)
        metrics["latent_ld"] = get_linear_disentanglement(truth, x)
    if config["eval"]["slot_ld"]:
        x = jnp.concatenate(slots, axis=0)
        metrics["slot_ld"] = get_linear_disentanglement(truth, x)
    if config["eval"]["mcc"]:
        x = jnp.concatenate(z, axis=0)
        mcc, pearson = get_mcc(truth, x)
        metrics["mcc"] = mcc
        metrics["pearson"] = pearson

    utl.save_metrics([metrics], run_name, "eval_metrics.csv")


def main():
    parser = argparse.ArgumentParser(
        prog="slotmachine", description="Train slotmachine autoencoder."
    )
    parser.add_argument("config", help="Model and training configuration.")
    parser.add_argument("-n", "--name", help="Name for the training run.")
    parser.add_argument("-d", "--data", help="Dataset path.")
    parser.add_argument("-b", "--base", help="Base model name.")
    parser.add_argument(
        "-e",
        "--base-epoch",
        help="Base model epoch. If not provided, latest epoch is selected.",
    )
    args = parser.parse_args()
    config = utl.load_config_by_name(args.config)

    if "seed" not in config:
        config["seed"] = np.random.randint(2**31)

    key = jax.random.PRNGKey(config["seed"])

    if args.name is None:
        if not "name" in config:
            raise "Provide a name for the run!"
    else:
        config["name"] = args.name

    if args.data is None:
        if not "data_path" in config:
            raise "Provide a dataset path!"
    else:
        config["data_path"] = args.data

    if args.base is None:
        if not "model" in config:
            raise "Provide a model definition or base model!"
    else:
        base_config = utl.load_config(f"out/{args.base}/config.toml")
        config["type"] = base_config["type"]
        config["model"] = base_config["model"]
        config["base_model"] = args.base
        if not args.base_epoch is None:
            config["base_epoch"] = args.base_epoch

    if not "type" in config:
        config["type"] = "slotmachine"

    print(f"Starting run {args.config}...")

    key, subkey = jax.random.split(key, 2)

    if config["type"] == "slotmachine":
        model = Slotmachine(key=subkey, **config["model"])
    else:
        raise "Invalid model type provided!"

    if "base_model" in config:
        if not "base_epoch" in config:
            config["base_epoch"] = utl.get_latest_epoch(args.base)
            if config["base_epoch"] == -1:
                raise "Base model has no available checkpoints!"

        print(
            f"Loading base model {config["base_model"]} epoch {config["base_epoch"]}..."
        )
        model = eqx.tree_deserialise_leaves(
            f"out/{config["base_model"]}/ckpt_{config["base_epoch"]}.eqx", model
        )

    if not os.path.exists("out"):
        os.mkdir("out")

    if os.path.exists(f"out/{config["name"]}"):
        run_index = 0
        while os.path.exists(f"out/{config["name"]}_{run_index}"):
            run_index += 1
        run_name = f"{config["name"]}_{run_index}"
    else:
        run_name = config["name"]

    if not "train" in config:
        sliding_window = False
    elif "sliding_window" in config["train"]:
        if "dis_loss" in config["train"] and config["train"]["dis_loss"]:
            raise "Sliding window training with disentanglement loss not supported!"
        sliding_window = config["train"]["sliding_window"]
    else:
        sliding_window = False

    print(f"Loading data...")
    data = utl.load_dataset(config["data_path"])

    length = config["model"]["seq_length"]
    if config["model"]["dynamics"]:
        length += 1

    print(f"Preprocessing data...")
    img_o, img_p, magnitudes, properties, actions = utl.preprocess_data(
        data, length, sliding_window
    )

    if "train" in config:
        dataset = jdl.ArrayDataset(img_o, img_p, magnitudes, properties, actions)
        key, subkey = jax.random.split(key, 2)
        dataloader = jdl.DataLoader(
            dataset,
            "jax",
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            drop_last=True,
            generator=subkey,
        )

    if "eval" in config:
        if config["eval"]["slot_ld"] or config["eval"]["latent_ld"]:
            masks = data["masks"]
        else:
            N = img_o.shape[0]
            masks = jnp.zeros((N, 0, 0, 0, 0), dtype=bool)

        dataset = jdl.ArrayDataset(
            img_o, img_p, magnitudes, properties, actions, masks, data["groundtruth_o"]
        )
        eval_dataloader = jdl.DataLoader(
            dataset, "jax", batch_size=config["eval"]["batch_size"]
        )

    print(f"Writing to output folder {run_name}...")
    os.mkdir(f"out/{run_name}")
    utl.save_config(config, f"out/{run_name}/config.toml")

    if "train" in config:
        print("Starting training...")
        key, subkey = jax.random.split(key, 2)
        train(model, dataloader, run_name, config, key=subkey)
    if "eval" in config:
        print("Running evaluation...")
        key, subkey = jax.random.split(key, 2)
        eval(model, eval_dataloader, run_name, config, key=subkey)


if __name__ == "__main__":
    main()

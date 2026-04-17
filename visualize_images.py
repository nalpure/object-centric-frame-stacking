import argparse
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

from slotmachine.model import Slotmachine
import slotmachine.utils as utl


def main():
    parser = argparse.ArgumentParser(
        prog="slotmachine", description="Visualize slotmachine reconstructions."
    )
    parser.add_argument("name", help="Name of trained model to load.", type=str)
    parser.add_argument("epoch", help="Epoch of trained model to load.", type=int)
    parser.add_argument("-d", "--data", help="Dataset path.")

    args = parser.parse_args()

    model_path = f"out/{args.name}/ckpt_{args.epoch}.eqx"

    config = utl.load_config(f"out/{args.name}/config.toml")

    key = jax.random.PRNGKey(config["seed"])

    key, subkey = jax.random.split(key, 2)
    model = Slotmachine(key=subkey, **config["model"])
    model = eqx.tree_deserialise_leaves(model_path, model)

    print(f"Loading data...")
    if not args.data is None:
        data = utl.load_dataset(args.data)
    else:
        data = utl.load_dataset(config["data_path"])

    img_o = utl.reorder_obs(data["img_o"]) / 255
    img_p = utl.reorder_obs(data["img_p"]) / 255
    seq_len_input = img_o.shape[1]

    if seq_len_input < model.seq_length:
        raise ValueError(
            f"Input sequence length {seq_len_input} is shorter than model sequence length {model.seq_length}."
        )
    elif seq_len_input > model.seq_length:
        print(
            f"Warning: Input sequence length {seq_len_input} is longer than model sequence length {model.seq_length}. Truncating input sequences."
        )
        img_o = img_o[:, : model.seq_length]
        img_p = img_p[:, : model.seq_length]

    magnitudes = data["magnitudes"]
    properties = data["properties"]
    actions = data["actions"]

    for index in [1, 2, 3, 4, 5]:
        if config["model"]["action_size"] == 0:
            action = None
        else:
            action = actions[index]

        key, subkey = jax.random.split(key, 2)
        _, z_o, _, recon, _, masks = model(
            img_o[index],
            config["slot"]["num_slots"],
            config["slot"]["sa_iterations"],
            subkey,
            None,
            action,
        )
        _, z_p, _, _, _, _ = model(
            img_p[index],
            config["slot"]["num_slots"],
            config["slot"]["sa_iterations"],
            subkey,
            None,
            action,
        )

        recon_loss = np.mean(np.square(recon - img_o[index]))

        print(f"Sample {index}:")
        print(f"\tReconstruction L2: {recon_loss}")
        print(f"\tPerturbed {properties[index]} by {magnitudes[index]}")
        idx = jnp.full((config["slot"]["num_slots"], 1), properties[index])
        print(f"z original: {jnp.take_along_axis(z_o, idx, axis=1)}")
        print(f"z perturbed: {jnp.take_along_axis(z_p, idx, axis=1)}")

        _, axes = plt.subplots(
            2 + config["slot"]["num_slots"],
            model.seq_length,
            figsize=(3 * model.seq_length, 15),
        )
        if model.seq_length == 1:
            axes = np.expand_dims(axes, axis=1)

        images = np.clip(np.transpose(img_o[index], (0, 2, 3, 1)), 0, 1)
        recons = np.clip(np.transpose(recon, (0, 2, 3, 1)), 0, 1)
        masks = np.transpose(masks, (0, 1, 3, 4, 2))

        for i in range(model.seq_length):
            axes[0, i].imshow(images[i])
            axes[0, i].set_title(f"t = {i}", fontsize=12, fontweight="bold")
            axes[0, i].axis("off")
            axes[1, i].axis("off")

            for j in range(config["slot"]["num_slots"]):
                axes[2 + j, i].imshow(masks[i, j])
                axes[2 + j, i].axis("off")

        plt.tight_layout()
        plt.savefig(f"out/{args.name}/epoch_{args.epoch}_sample_{index}.png")
        plt.close()


if __name__ == "__main__":
    main()

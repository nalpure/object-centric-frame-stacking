import os
import tomllib
import csv
import matplotlib.pyplot as plt
import numpy as np
import re


def get_latest_epoch(run_name):
    base_path = "out" + f"/{run_name}"
    pattern = re.compile(r"ckpt_(\d+)\.eqx$")

    max_index = -1

    for file in os.listdir(base_path):
        match = pattern.match(file)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index

    return max_index


def get_full_run_chain(run_name, base_dir, is_dis):
    """
    Given a run name, follow the chain of base models until the original base run.
    Returns a list of run names in chronological order (base first, then continuation).
    """
    chain = []
    current = run_name
    offset = 0
    base_epoch = 0
    while current:
        config_path = os.path.join(base_dir, current, "config.toml")
        if not os.path.exists(config_path):
            print("Path not found!")
            break
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        if is_dis and not config["train"]["dis_loss"]:
            offset += base_epoch
        else:
            chain.insert(0, current)
        current = config.get("base_model")  # None if not a continuation
        epoch = config.get("base_epoch")
        if epoch:
            base_epoch = int(epoch)
        elif current:
            base_epoch = get_latest_epoch(current)
    return chain, offset


def load_reconstruction_losses(run_chain, base_dir):
    """
    Given a list of runs (in order), concatenate their reconstruction losses.
    Uses only the built-in csv module.
    """
    losses = []
    for run in run_chain:
        csv_path = os.path.join(base_dir, run, "train_metrics.csv")
        print(csv_path)
        if not os.path.exists(csv_path):
            print(f"Warning: missing train_metrics.csv in {run}")
            continue
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if "reconstruction_loss" not in reader.fieldnames:
                print(f"Warning: 'reconstruction_loss' not found in {run}")
                continue
            for row in reader:
                try:
                    loss = float(row["reconstruction_loss"])
                    losses.append(loss)
                except (ValueError, KeyError):
                    continue
    return losses


def plot_stitched_losses(
    run_names, clear_names, dis_names, small, base_dir, save_path=None
):
    """
    For each run name, follow the base chain, stitch losses, and plot.
    """
    plt.figure(figsize=(10, 6))

    for i, run_name in enumerate(run_names):
        if small[i]:
            step = 0.25
        else:
            step = 1

        run_chain, _ = get_full_run_chain(run_name, base_dir, False)
        losses = load_reconstruction_losses(run_chain, base_dir)
        plt.plot(
            np.arange(step, step * (1 + len(losses)), step),
            losses,
            label=f"{clear_names[i]}",
            color=f"C{2*i}",
        )

        run_chain, offset = get_full_run_chain(dis_names[i], base_dir, True)
        losses = load_reconstruction_losses(run_chain, base_dir)
        plt.plot(
            np.arange(step * (1 + offset), step * (1 + offset + len(losses)), step),
            losses,
            color=f"C{2*i+1}",
            linestyle="--",
            label=f"{clear_names[i]} Dis",
        )

    plt.xlabel("80k Sample Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Stitched Reconstruction Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)


# === Example usage ===
# Compare several runs, stitching each with its base chain
run_names = [
    # "4b_4c_fin",
    # "4b_4c_entangle_fin",
    "4b_4c_leftpos_fin_0",
    # "4b_4c_leftvel_fin_2",
    "4b_4c_leftpos_fin_0",
    # "4b_4c_leftvel_0",
    # "4b_4c_0",
    # "4b_4c_early_dis_leftpos_0",
]
clear_names = [
    # "Standard",
    # "Entangled",
    "Leftpos",
    # "Leftvel",
    # "Old Leftpos",
    # "Old Leftvel",
    # "Old Standard",
    "Early Leftpos",
]
dis_names = [
    # "4b_4c_dis_fin",
    # "4b_4c_entangle_dis_fin",
    "4b_4c_leftpos_dis_fin_0",
    # "4b_4c_leftvel_dis_fin",
    # "4b_4c_leftpos_dis_0",
    # "4b_4c_leftvel_dis_0",
    # "4b_4c_dis_1",
    "4b_4c_early_dis_leftpos_0",
]
small = [False, False]  # , False, True, True]  # False, #False,
plot_stitched_losses(
    run_names, clear_names, dis_names, small, base_dir="out", save_path="test.png"
)

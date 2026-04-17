import numpy as np
import tomllib
import tomli_w
import os
import re
import csv
from importlib import resources


def load_config_by_name(name):
    try:
        path = resources.files("slotmachine").joinpath("../../cfgs", name)

        with path.open("rb") as f:
            config = tomllib.load(f)

        return config
    except FileNotFoundError:
        print(f"Config file '{name}' does not exist!")
        raise
    except Exception as e:
        print(f"Error occured while loading config: {e}")
        raise


def load_config(path):
    with open(path, "rb") as f:
        config = tomllib.load(f)

    return config


def save_config(config, path):
    with open(path, "wb") as f:
        tomli_w.dump(config, f)


def load_dataset(path):
    dataset = np.load(path, mmap_mode="r")

    img_o = dataset["img_o"]
    img_p = dataset["img_p"]

    groundtruth_o = dataset["groundtruth_o"]
    groundtruth_p = dataset["groundtruth_p"]
    magnitudes = dataset["magnitudes"]
    indices = dataset["indices"]
    properties = dataset["properties"]
    if "actions" in dataset.keys():
        actions = dataset["actions"]
    else:
        actions = np.zeros((len(img_o), 1))
    if "masks_o" in dataset.keys():
        masks = dataset["masks_o"]
    else:
        masks = np.zeros((len(img_o), 1))

    dataset.close()

    return dict(
        img_o=img_o,
        img_p=img_p,
        groundtruth_o=groundtruth_o,
        groundtruth_p=groundtruth_p,
        magnitudes=magnitudes,
        indices=indices,
        properties=properties,
        actions=actions,
        masks=masks,
    )


def aggregate_metrics(agg, new):
    for key, value in new.items():
        if key in agg:
            agg[key] += value
        else:
            agg[key] = value


def normalize_metrics(metrics, n):
    for key in metrics.keys():
        metrics[key] /= n


def save_metrics(metrics, run_name, file_name):
    out_path = "out" + f"/{run_name}/{file_name}"
    if not os.path.exists(out_path):
        with open(out_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            writer.writeheader()
            for m in metrics:
                writer.writerow(m)
    else:
        with open(out_path, "a") as f:
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            for m in metrics:
                writer.writerow(m)


def get_latest_epoch(base_name):
    base_path = "out" + f"/{base_name}"
    pattern = re.compile(r"ckpt_(\d+)\.eqx$")

    max_index = -1

    for file in os.listdir(base_path):
        match = pattern.match(file)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index

    return max_index


def reorder_obs(o):
    o = np.transpose(
        o, (0, 1, 4, 2, 3)
    )  # [dataset_size sequence_length 3 width height]
    return o


def _sliding_windows_axis1(arr, window_size):
    """
    Create overlapping windows along axis=1 for an array shaped (N, T, ...).
    Returns array shaped (N*(T-window+1), window, ...).
    Uses sliding_window_view when available; falls back to safe Python loop.
    """
    N, T = arr.shape[0], arr.shape[1]
    if T < window_size:
        raise ValueError(f"Sequence length T={T} smaller than requested window_size={window_size}")

    n_windows = T - window_size + 1
    sw = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window_size, axis=1)
    # sw shape: (N, n_windows, C, H, W, window_size)
    sw = np.moveaxis(sw, -1, 2)
    # sw shape: (N, n_windows, window_size, C, H, W)
    out = sw.reshape(N * n_windows, window_size, *arr.shape[2:])
    return np.ascontiguousarray(out)
    

def preprocess_data(data, seq_length, sliding_window=False):
    img_o = reorder_obs(data["img_o"])  # expected shape (N, T, C, H, W)
    img_p = reorder_obs(data["img_p"])
    shape_orig = img_o.shape
    T = img_o.shape[1]

    if not sliding_window:
        # Strict cropping: keep first seq_length timesteps
        if T < seq_length:
            raise ValueError(f"Dataset sequence length T={T} is shorter than requested seq_length={seq_length}.")

        img_o = np.ascontiguousarray(img_o[:, :seq_length])
        img_p = np.ascontiguousarray(img_p[:, :seq_length])
        magnitudes = data["magnitudes"]
        properties = data["properties"]
        actions = data["actions"]
        print(f"Applied strict cropping: original shape {shape_orig}, new shape {img_o.shape}")
    else:
        # Sliding windows: produce windows for image arrays
        img_o = _sliding_windows_axis1(img_o, seq_length)  # (N*n_windows, seq_length, C, H, W)
        img_p = _sliding_windows_axis1(img_p, seq_length)

        num_windows = T - seq_length + 1
        magnitudes = np.repeat(data["magnitudes"], num_windows, axis=0)
        properties = np.repeat(data["properties"], num_windows, axis=0)
        actions = np.repeat(data["actions"], num_windows, axis=0)
        print(f"Created sliding windows: original shape {shape_orig}, new shape {img_o.shape}")

    return img_o, img_p, magnitudes, properties, actions
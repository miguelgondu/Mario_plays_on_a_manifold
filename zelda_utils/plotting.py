"""
This script defines some functions that
plot levels.
"""
import os
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL

filepath = Path(__file__).parent.resolve()
Tensor = torch.Tensor


def absolute(path_str):
    return str(Path(path_str).absolute())


encoding = {
    "w": 0,
    "A": 1,
    ".": 2,
    "g": 3,
    "1": 4,
    "2": 5,
    "3": 6,
    "+": 7,
}

sprites = {
    encoding["w"]: absolute(f"{filepath}/sprites/w.png"),
    encoding["A"]: absolute(f"{filepath}/sprites/A.png"),
    encoding["."]: absolute(f"{filepath}/sprites/f.png"),
    encoding["g"]: absolute(f"{filepath}/sprites/g.png"),
    encoding["1"]: absolute(f"{filepath}/sprites/1.png"),
    encoding["2"]: absolute(f"{filepath}/sprites/2.png"),
    encoding["3"]: absolute(f"{filepath}/sprites/3.png"),
    encoding["+"]: absolute(f"{filepath}/sprites/+.png"),
}


def save_level_from_array(path, level, title=None, dpi=150):
    # Assuming that the level is a bunch of classes.
    image = get_img_from_level(level)
    plt.imshow(255 * np.ones_like(image))  # White background
    plt.imshow(image)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close()


def plot_level_from_array(ax, level, title=None):
    # Assuming that the level is a bunch of classes.
    image = get_img_from_level(level)
    ax.imshow(255 * np.ones_like(image))  # White background
    ax.imshow(image)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)


def get_img_from_level(level: np.ndarray):
    image = []
    for row in level:
        image_row = []
        for c in row:
            tile = np.asarray(PIL.Image.open(sprites[c]).convert("RGB")).astype(int)
            image_row.append(tile)
        image.append(image_row)

    image = [np.hstack([tile for tile in row]) for row in image]
    image = np.vstack([np.asarray(row) for row in image])

    return image


def plot_level_from_decoded_tensor(dec: Tensor, ax):
    """
    Plots decoded tensor as level in ax.
    Expects {dec} to have a batch component.
    """
    level = dec.argmax(dim=1).detach().numpy()
    plot_level_from_array(ax, level)

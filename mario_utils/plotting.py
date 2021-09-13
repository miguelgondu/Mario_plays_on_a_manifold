"""
This script defines some functions that
plot levels.
"""
from mario_utils.levels import onehot_to_levels
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL

# SPRITES_PATH = os.environ.get("SPRITES_PATH")
# ENCODING_PATH = os.environ.get("ENCODING_PATH")

Tensor = torch.Tensor
SPRITES_PATH = (
    "/home/miguel/Projects/MarioVAE/sprites.json"
)
ENCODING_PATH = (
    "/home/miguel/Projects/MarioVAE/encoding.json"
)

with open(SPRITES_PATH) as fp:
    sprites = json.load(fp)

with open(ENCODING_PATH) as fp:
    encoding = json.load(fp)


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
            if c == encoding["-"]:  # There must be a smarter way than hardcoding this.
                # white background
                tile = (255 * np.ones((16, 16, 3))).astype(int)
            elif c == -1:
                # masked
                tile = (128 * np.ones((16, 16, 3))).astype(int)
            else:
                tile = np.asarray(
                    PIL.Image.open(sprites[str(c)]).convert("RGB")
                ).astype(int)
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
    level = onehot_to_levels(dec.detach().numpy())[0]
    plot_level_from_array(ax, level)

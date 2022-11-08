"""
Plots four things: A grid of levels in latent space,
a playable level, a non-playable level, and the ground truth mask.
"""

from pathlib import Path

import torch as t
import matplotlib.pyplot as plt
import numpy as np
from geometries import BaselineGeometry

from vae_models.vae_mario_hierarchical import VAEMarioHierarchical
from utils.experiment import load_csv_as_map

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 16

plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=SMALL_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("figure", titlesize=BIGGER_SIZE)

ROOT_DIR = Path(__file__).parent.parent.resolve()
vae_path = ROOT_DIR / "trained_models" / "ten_vaes" / "vae_mario_hierarchical_id_1.pt"
path_to_gt = (
    ROOT_DIR
    / "data"
    / "array_simulation_results"
    / "ten_vaes"
    / "ground_truth"
    / f"{vae_path.stem}.csv"
)

p_map = load_csv_as_map(path_to_gt)
bg = BaselineGeometry(p_map, "baseline_for_plotting_journal_version", vae_path)

vae = VAEMarioHierarchical()
vae.load_state_dict(t.load(vae_path, map_location=vae.device))


def plot_grid_and_levels():
    PLOTS_PATH = ROOT_DIR / "data" / "plots" / "journal_version" / "banner"
    PLOTS_PATH.mkdir(exist_ok=True, parents=True)

    # Plotting the grid of levels
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    _, imgs = vae.plot_grid(n_rows=10, n_cols=10, ax=ax, return_imgs=True)
    # ax.set_title("Latent Space of Super Mario Bros", fontsize=BIGGER_SIZE)
    ax.axis("off")

    layer_on_top = np.where(bg.grid == 1.0, np.NaN, bg.grid)
    ax.imshow(
        layer_on_top,
        extent=[-5, 5, -5, 5],
        alpha=0.4,
        cmap="autumn",
        vmin=0.0,
        vmax=1.0,
    )
    fig.savefig(
        PLOTS_PATH / "banner_grid_w_mask_journal_version.png",
        dpi=120,
        bbox_inches="tight",
    )

    non_playable = imgs[54]
    playable = imgs[67]

    # plotting the nonplayable one
    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 7))
    ax1.imshow(non_playable)
    ax1.axis("off")
    # ax1.set_title("Not functional", fontsize=BIGGER_SIZE)
    fig1.savefig(
        PLOTS_PATH / "banner_not_playable_journal_version.png",
        dpi=100,
        bbox_inches="tight",
    )

    # plotting the playable one
    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7))
    ax2.imshow(playable)
    ax2.axis("off")
    # ax2.set_title("Functional", fontsize=BIGGER_SIZE)
    fig2.savefig(
        "./data/plots/ten_vaes/paper_ready/banner_playable.png",
        dpi=100,
        bbox_inches="tight",
    )


def plot_all_levels():
    _, imgs = vae.plot_grid(return_imgs=True)
    for i, img in enumerate(imgs):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img)
        ax.axis("off")
        fig.savefig(f"./data/plots/ten_vaes/grids/all_levels/{i:04d}.png")
        plt.close(fig)


if __name__ == "__main__":
    plot_grid_and_levels()

"""
Plots four things: A grid of levels in latent space,
a playable level, a non-playable level, and the ground truth mask.
"""

from pathlib import Path

import torch as t
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from geometries import BaselineGeometry, DiscretizedGeometry

from vae_models.vae_mario_hierarchical import VAEMarioHierarchical
from utils.experiment import load_csv_as_map
from utils.mario.plotting import get_img_from_level, save_level_from_array

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

MODEL_ID = 2

vae_path = Path(f"./trained_models/ten_vaes/vae_mario_hierarchical_id_{MODEL_ID}.pt")
path_to_gt = (
    Path("./data/array_simulation_results/ten_vaes/ground_truth")
    / f"{vae_path.stem}.csv"
)

p_map = load_csv_as_map(path_to_gt)
bg = BaselineGeometry(p_map, "baseline_for_plotting_journal_version", vae_path)
dg = DiscretizedGeometry(
    p_map, "geometry_for_plotting_banner_journal_version", vae_path, force=True
)

# Getting more levels for plotting.
bg.interpolation.n_points_in_line = 100
dg.interpolation.n_points_in_line = 100

vae = VAEMarioHierarchical()
vae.load_state_dict(t.load(vae_path, map_location=vae.device))


def save_img(img, name):
    IMGS_PATH = ROOT_DIR / "data" / "plots" / "journal_version" / "all_levels"
    IMGS_PATH.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(1, 1)
    ax.axis("off")
    ax.imshow(img)
    fig.savefig(IMGS_PATH / "data" f"{name}.png")
    plt.close(fig)


def plot_grid_and_levels(save_images: bool = True):
    PLOTS_PATH = ROOT_DIR / "data" / "plots" / "journal_version" / "banner"
    PLOTS_PATH.mkdir(exist_ok=True, parents=True)

    # Plotting the grid of levels
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    _, imgs = vae.plot_grid(n_rows=20, n_cols=20, ax=ax, return_imgs=True)
    # if save_images:
    #     for i, img in imgs:
    #         save_img(img, f"{i:05d}")

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

    z = t.Tensor([-4.5, -4.5])
    z_prime = t.Tensor([-4.5, -0.8])
    res = dg.interpolation._full_interpolation(z, z_prime).detach().numpy()
    linear_interpolation, levels = bg.interpolate(z, z_prime)
    linear_interpolation = linear_interpolation.detach().numpy()
    geodesic_color = "#F2BB05"
    ax.plot(
        res[:, 0],
        res[:, 1],
        lw=3,
        label="Ours",
        c=geodesic_color,
        path_effects=[pe.Stroke(linewidth=5, foreground="k"), pe.Normal()],
    )
    linear_color = "#2A94DF"
    ax.plot(
        linear_interpolation[:, 0],
        linear_interpolation[:, 1],
        lw=3,
        label="Linear",
        c=linear_color,
        path_effects=[pe.Stroke(linewidth=5, foreground="k"), pe.Normal()],
    )
    ax.scatter(
        linear_interpolation[[0, -1, 50], 0],
        linear_interpolation[[0, -1, 50], 1],
        s=120,
        c=[geodesic_color, geodesic_color, linear_color],
        edgecolors="k",
        zorder=3,
    )
    ax.set_xlim((-5, 5))
    ax.set_ylim((-5, 5))
    # handles, labels = plt.gca().get_legend_handles_labels()
    # red_patch = mpatches.Patch(color="red", label="Non-functional")
    # handles.extend([red_patch])
    # plt.legend(handles=[red_patch])
    ax.legend(prop={"size": 12})
    fig.savefig(
        PLOTS_PATH / "banner_grid_w_mask_camera_ready.png",
        dpi=120,
        bbox_inches="tight",
    )
    # plt.show()
    # plt.close(fig)
    # save_levels(levels.detach().numpy())

    # Lvl 78 is a good example of non-solvable.
    imgs = [get_img_from_level(lvl.detach().numpy()) for lvl in levels]
    if save_images:
        for i, img in enumerate(imgs):
            save_img(img, f"{i:05}")

    beginning = imgs[0]
    end = imgs[-1]

    # # # plotting the nonplayable one
    # fig1, ax1 = plt.subplots(1, 1, figsize=(7, 7))
    # ax1.imshow(beginning)
    # ax1.axis("off")
    # # ax1.set_title("Not functional", fontsize=BIGGER_SIZE)
    # # fig1.savefig(
    # #     "./data/plots/ten_vaes/paper_ready/banner_not_playable.png",
    # #     dpi=100,
    # #     bbox_inches="tight",
    # # )

    # # plotting the playable one
    # fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7))
    # ax2.imshow(end)
    # ax2.axis("off")
    # # ax2.set_title("Functional", fontsize=BIGGER_SIZE)
    # # fig2.savefig(
    # #     "./data/plots/ten_vaes/paper_ready/banner_playable.png",
    # #     dpi=100,
    # #     bbox_inches="tight",
    # # )

    plt.show()
    plt.close()


def save_levels(levels: np.ndarray):
    for i, lvl in enumerate(levels):
        save_level_from_array(f"./data/levels_banner_{i:05d}.png", lvl)


# def plot_all_levels():
#     _, imgs = vae.plot_grid(return_imgs=True)
#     for i, img in enumerate(imgs):
#         fig, ax = plt.subplots(1, 1)
#         ax.imshow(img)
#         ax.axis("off")
#         fig.savefig(f"./data/plots/ten_vaes/grids/all_levels/{i:04d}.png")
#         plt.close(fig)


if __name__ == "__main__":
    plot_grid_and_levels(save_images=False)

"""
Plots four things: A grid of levels in latent space,
a playable level, a non-playable level, and the ground truth mask.
"""

from pathlib import Path

import torch as t
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from geometry import BaselineGeometry, DiscretizedGeometry

from vae_mario_hierarchical import VAEMarioHierarchical
from experiment_utils import load_csv_as_map
from mario_utils.plotting import get_img_from_level, save_level_from_array

from simulator import test_level_from_decoded_tensor

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


vae_path = Path("./models/ten_vaes/vae_mario_hierarchical_id_0.pt")
path_to_gt = (
    Path("./data/array_simulation_results/ten_vaes/ground_truth")
    / f"{vae_path.stem}.csv"
)

p_map = load_csv_as_map(path_to_gt)
bg = BaselineGeometry(p_map, "baseline_for_plotting", vae_path)
dg = DiscretizedGeometry(p_map, "geometry_for_plotting_banner", vae_path)

# Getting more levels for plotting.
bg.interpolation.n_points_in_line = 100
dg.interpolation.n_points_in_line = 100

vae = VAEMarioHierarchical()
vae.load_state_dict(t.load(vae_path, map_location=vae.device))
vae.eval()


def all_animations():
    # Plotting the grid of levels
    # fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    # ax.set_title("Latent Space of Super Mario Bros", fontsize=BIGGER_SIZE)
    # ax.axis("off")

    layer_on_top = np.where(bg.grid == 1.0, np.NaN, bg.grid)
    # ax.imshow(
    #     layer_on_top,
    #     extent=[-5, 5, -5, 5],
    #     alpha=0.4,
    #     cmap="autumn",
    #     vmin=0.0,
    #     vmax=1.0,
    # )

    z = t.Tensor([4.8, 3.0])
    z_prime = t.Tensor([3.0, 3.6])
    full_geodesic_interpolation = (
        dg.interpolation._full_interpolation(z, z_prime).detach().numpy()
    )
    geodesic_interpolation, levels_geodesic = dg.interpolate(z, z_prime)
    linear_interpolation, levels_linear = bg.interpolate(z, z_prime)
    linear_interpolation = linear_interpolation.detach().numpy()

    # First half of the linear interpolation
    save_path = "./data/plots/animation_for_twitter/linear_interpolation_first_half/"
    animation_linear_interpolation_from(
        1,
        len(linear_interpolation) // 2,
        linear_interpolation,
        layer_on_top,
        save_path=save_path,
    )

    # Second half of the linear interpolation
    save_path = "./data/plots/animation_for_twitter/linear_interpolation_second_half/"
    animation_linear_interpolation_from(
        len(linear_interpolation) // 2,
        len(linear_interpolation),
        linear_interpolation,
        layer_on_top,
        save_path=save_path,
    )

    # geodesic_color = "#F2BB05"
    # ax.plot(
    #     full_geodesic_interpolation[:, 0],
    #     full_geodesic_interpolation[:, 1],
    #     lw=3,
    #     label="Ours",
    #     c=geodesic_color,
    #     path_effects=[pe.Stroke(linewidth=5, foreground="k"), pe.Normal()],
    # )
    # linear_color = "#2A94DF"
    # ax.plot(
    #     linear_interpolation[:, 0],
    #     linear_interpolation[:, 1],
    #     lw=3,
    #     label="Linear",
    #     c=linear_color,
    #     path_effects=[pe.Stroke(linewidth=5, foreground="k"), pe.Normal()],
    # )

    # Beginning and ending points
    # ax.scatter(
    #     linear_interpolation[[0, -1], 0],
    #     linear_interpolation[[0, -1], 1],
    #     s=120,
    #     c="black",
    #     edgecolors="k",
    #     zorder=3,
    # )

    # Linear interpolation's problematic point
    idx_problematic = 50
    # ax.scatter(
    #     linear_interpolation[[idx_problematic], 0],
    #     linear_interpolation[[idx_problematic], 1],
    #     s=120,
    #     c=linear_color,
    #     edgecolors="k",
    #     zorder=3,
    # )

    # Level thar geodesic doesn't struggle with.
    # ax.scatter(
    #     geodesic_interpolation[[50], 0],
    #     geodesic_interpolation[[50], 1],
    #     s=120,
    #     c=geodesic_color,
    #     edgecolors="k",
    #     zorder=3,
    # )
    # ax.set_xlim((2.5, 5.0))
    # ax.set_ylim((2.5, 4.2))
    # ax.legend(prop={"size": 12})
    # fig.savefig(
    #     "./data/plots/ten_vaes/paper_ready/animation_for_twitter_base.png",
    #     dpi=120,
    #     bbox_inches="tight",
    # )

    # Simulating the problematic level for the line
    # test_level_from_decoded_tensor(levels_linear[idx_problematic], visualize=True)

    # Simulating the good level for the geodesic
    # test_level_from_decoded_tensor(levels_geodesic[50], visualize=True)

    plt.show()
    plt.close()

    # plot_all_levels()


def animation_linear_interpolation_from(
    beginning: int,
    end: int,
    linear_interpolation: t.Tensor,
    base_image: np.ndarray,
    save_path: Path,
):
    linear_color = "#2A94DF"
    for i in range(beginning, end):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.axis("off")

        line_to_plot = linear_interpolation[:i]

        # Plotting the background
        _, _ = vae.plot_grid(n_rows=20, n_cols=20, ax=ax, return_imgs=True)
        ax.imshow(
            base_image,
            extent=[-5, 5, -5, 5],
            alpha=0.4,
            cmap="autumn",
            vmin=0.0,
            vmax=1.0,
        )

        # Starting and ending points
        ax.scatter(
            linear_interpolation[[0, -1], 0],
            linear_interpolation[[0, -1], 1],
            s=120,
            c="black",
            edgecolors="k",
            zorder=3,
        )

        # The line
        ax.plot(
            line_to_plot[:, 0],
            line_to_plot[:, 1],
            lw=3,
            label="Linear",
            c=linear_color,
            path_effects=[pe.Stroke(linewidth=5, foreground="k"), pe.Normal()],
        )

        # Final point
        ax.scatter(
            line_to_plot[[-1], 0],
            line_to_plot[[-1], 1],
            s=120,
            c=linear_color,
            edgecolors="k",
            zorder=3,
        )

        ax.set_xlim((2.5, 5.0))
        ax.set_ylim((2.5, 4.2))

        fig.savefig(save_path / f"{i:05d}.png")
        plt.close(fig)


def plot_all_levels():
    _, imgs = vae.plot_grid(return_imgs=True)
    for i, img in enumerate(imgs):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img)
        ax.axis("off")
        fig.savefig(f"./data/plots/animation_for_twitter/levels_animation_{i:05d}.png")
        plt.close(fig)


if __name__ == "__main__":
    all_animations()
    # animation_linear_interpolation_first_half()
from pathlib import Path

import torch as t
import matplotlib.pyplot as plt
import numpy as np
from geometries import BaselineGeometry, DiscretizedGeometry

from vae_mario_hierarchical import VAEMarioHierarchical
from vae_mario_obstacles import VAEWithObstacles
from utils.experiment import grid_from_map, load_csv_as_map

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


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

vae = VAEMarioHierarchical()
vae.load_state_dict(t.load(vae_path, map_location=vae.device))


def plot_latent_spaces():
    strict_p_map = {z: 1.0 if p == 1.0 else 0.0 for z, p in p_map.items()}
    ddg = DiscretizedGeometry(
        strict_p_map,
        "ddg_without_calibrating",
        vae_path,
        n_grid=100,
        with_obstacles=False,
        x_lims=(-5, 5),
        y_lims=(-5, 5),
        force=True,
    )

    fig1, ax1 = plt.subplots(1, 1, figsize=(7, 7))
    ax1.imshow(ddg.grid, cmap="viridis")
    # plt.show()

    ddg2 = DiscretizedGeometry(
        strict_p_map,
        "ddg_with_calibrating",
        vae_path,
        n_grid=100,
        x_lims=(-5, 5),
        y_lims=(-5, 5),
        force=True,
    )

    fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7))
    ax2.imshow(ddg2.grid, cmap="viridis")
    plt.show()


def load_and_plot_metric_volumes():
    # _, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * 7, 1 * 7))
    a_calibrated = np.load(
        "./data/processed/metric_volumes/ddg_with_calibrating/vae_mario_hierarchical_id_0.npz"
    )
    a_uncalibrated = np.load(
        "./data/processed/metric_volumes/ddg_without_calibrating/vae_mario_hierarchical_id_0.npz"
    )

    zs = a_calibrated["zs"]
    mv_calibrated = a_calibrated["metric_volumes"]
    mv_uncalibrated = a_uncalibrated["metric_volumes"]

    map_calibrated = {tuple(z.tolist()): mv for z, mv in zip(zs, mv_calibrated)}
    map_uncalibrated = {tuple(z.tolist()): mv for z, mv in zip(zs, mv_uncalibrated)}

    v_max = max(mv_calibrated.max(), mv_uncalibrated.max())
    v_min = min(mv_calibrated.min(), mv_uncalibrated.min())

    strict_p_map = {z: 1.0 if p == 1.0 else 0.0 for z, p in p_map.items()}
    obstacles = np.array([z for z, p in strict_p_map.items() if p == 0.0])

    gc = grid_from_map(map_calibrated)
    gu = grid_from_map(map_uncalibrated)

    # Set up figure and image grid
    fig = plt.figure(figsize=(7 * 2 + 0.15, 1 * 7))

    grid = ImageGrid(
        fig,
        111,  # as in plt.subplot(111)
        nrows_ncols=(1, 2),
        axes_pad=0.15,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="4%",
        cbar_pad=0.15,
    )

    # Add data to image grid
    # for ax in grid:
    #     im = ax.imshow(np.random.random((10,10)), vmin=0, vmax=1)
    ax1, ax2 = grid

    # Colorbar

    # plt.tight_layout()    # Works, but may still require rect paramater to keep colorbar labels visible
    # plt.show()
    ax1.imshow(gu, extent=[-5, 5, -5, 5], vmin=v_min, vmax=v_max, cmap="viridis")
    ax1.set_title("Uncalibrated", fontsize=BIGGER_SIZE)
    ax1.axis("off")
    plot = ax2.imshow(gc, extent=[-5, 5, -5, 5], vmin=v_min, vmax=v_max, cmap="viridis")
    # plt.colorbar(plot, ax=ax2)
    # ax2.scatter(obstacles[:, 0], obstacles[:, 1], s=5, marker="x", c="k")
    ax2.set_title("Calibrated", fontsize=BIGGER_SIZE)
    ax2.axis("off")

    cbar = ax2.cax.colorbar(plot)
    # cbar.set_ticks([])
    ax2.cax.toggle_label(True)
    # ax2.cax.set_ylabel(r"$\log(\det(M(z))$", rotation=270, fontsize=BIGGER_SIZE)
    # ax2.cax.set_yticks([])

    # plt.tight_layout()
    fig.savefig(
        "./data/plots/ten_vaes/paper_ready/calibrating_effects.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    # plot_latent_spaces()
    load_and_plot_metric_volumes()

"""
TODO: For each experiment get
- a p_map plot
- the approximated manifold (using ddg)
- example interpolations
- example diffusions.
"""
from pathlib import Path

import torch as t
import matplotlib.pyplot as plt
import numpy as np
from geometries import BaselineGeometry, DiscretizedGeometry
from plotting_scripts.plot_before_and_after_calibrating import BIGGER_SIZE

from vae_models.vae_mario_hierarchical import VAEMarioHierarchical
from vae_models.vae_mario_obstacles import VAEWithObstacles
from utils.experiment import grid_from_map, load_arrays_as_map, load_csv_as_map
from vae_models.vae_zelda_hierachical import VAEZeldaHierarchical


def plot_example_of_approximation():
    vae_path = Path("./models/ten_vaes/vae_mario_hierarchical_id_0.pt")
    path_to_gt = (
        Path("./data/array_simulation_results/ten_vaes/ground_truth")
        / f"{vae_path.stem}.csv"
    )
    p_map = load_csv_as_map(path_to_gt)
    obstacles = grid_from_map(p_map)

    dg = DiscretizedGeometry(p_map, "geometry_for_plotting_animations", vae_path)
    approximation = dg.grid

    vae = VAEMarioHierarchical()
    vae.load_state_dict(t.load(vae_path, map_location=vae.device))
    vae.eval()
    levels_grid = vae.plot_grid()

    zs_mv = dg.zs_of_metric_volumes
    mv = dg.metric_volumes
    map_mv = {tuple(z.tolist()): mv for z, mv in zip(zs_mv, mv)}
    calibrated = grid_from_map(map_mv)

    # vae = VAEZeldaHierarchical()
    # vae.load_state_dict(t.load(vae_path, map_location=vae.device))
    # levels_grid = vae.plot_grid(n_rows=6, n_cols=6, plot_all_levels=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(obstacles, cmap="Blues", extent=[-5, 5, -5, 5])
    # ax.set_title("Functional regions", fontsize=BIGGER_SIZE)
    img_paths = Path("./data/plots/animation_for_twitter/example_approximation_mario")
    img_paths.mkdir(exist_ok=True)

    ax.axis("off")
    fig.savefig(
        img_paths / "obstacles.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot = ax.imshow(calibrated, cmap="viridis", extent=[-5, 5, -5, 5])
    cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04, aspect=20)
    cbar.set_ticks([])
    # ax.set_title("Metric volume", fontsize=BIGGER_SIZE)
    ax.axis("off")
    fig.savefig(
        img_paths / "metric_volume.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot = ax.imshow(approximation, cmap="Blues", extent=[-5, 5, -5, 5])
    # plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
    # ax.set_title("Graph approximation", fontsize=BIGGER_SIZE)
    ax.axis("off")
    fig.savefig(
        img_paths / "approximation.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot = ax.imshow(levels_grid, extent=[-5, 5, -5, 5])
    # plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
    # ax.set_title("Decoded levels", fontsize=BIGGER_SIZE)
    ax.axis("off")
    fig.savefig(
        img_paths / "grid.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    plot_example_of_approximation()

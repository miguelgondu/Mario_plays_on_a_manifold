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
from geometry import BaselineGeometry, DiscretizedGeometry
from plotting_scripts.plot_before_and_after_calibrating import BIGGER_SIZE

from vae_mario_hierarchical import VAEMarioHierarchical
from vae_mario_obstacles import VAEWithObstacles
from experiment_utils import grid_from_map, load_arrays_as_map, load_csv_as_map
from vae_zelda_hierachical import VAEZeldaHierarchical


def plot_example():
    vae_path = Path("./models/zelda/zelda_hierarchical_final_0.pt")
    a = np.load(f"./data/processed/zelda/grammar_checks/{vae_path.stem}.npz")
    zs = a["zs"]
    p = a["playabilities"]
    p_map = load_arrays_as_map(zs, p)
    obstacles = grid_from_map(p_map)

    ddg = DiscretizedGeometry(
        p_map,
        "zelda_ddg_for_plotting",
        vae_path,
        n_grid=100,
        x_lims=(-4, 4),
        y_lims=(-4, 4),
        # force=True,
    )
    approximation = ddg.grid

    a_mv = np.load(
        f"./data/processed/metric_volumes/zelda_ddg_for_plotting/{vae_path.stem}.npz"
    )
    zs_mv = a_mv["zs"]
    mv = a_mv["metric_volumes"]
    map_mv = {tuple(z.tolist()): mv for z, mv in zip(zs_mv, mv)}
    calibrated = grid_from_map(map_mv)

    vae = VAEZeldaHierarchical()
    vae.load_state_dict(t.load(vae_path, map_location=vae.device))
    levels_grid = vae.plot_grid(n_rows=6, n_cols=6, plot_all_levels=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(obstacles, cmap="Blues", extent=[-4, 4, -4, 4])
    # ax.set_title("Functional regions", fontsize=BIGGER_SIZE)
    ax.axis("off")
    fig.savefig(
        "./data/plots/ten_vaes/paper_ready/example_approximation/obstacles.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot = ax.imshow(calibrated, cmap="viridis", extent=[-4, 4, -4, 4])
    cbar = plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04, aspect=20)
    cbar.set_ticks([])
    # ax.set_title("Metric volume", fontsize=BIGGER_SIZE)
    ax.axis("off")
    fig.savefig(
        "./data/plots/ten_vaes/paper_ready/example_approximation/metric_volume.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot = ax.imshow(approximation, cmap="Blues", extent=[-4, 4, -4, 4])
    # plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
    # ax.set_title("Graph approximation", fontsize=BIGGER_SIZE)
    ax.axis("off")
    fig.savefig(
        "./data/plots/ten_vaes/paper_ready/example_approximation/approximation.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot = ax.imshow(levels_grid, extent=[-4, 4, -4, 4])
    # plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
    # ax.set_title("Decoded levels", fontsize=BIGGER_SIZE)
    ax.axis("off")
    fig.savefig(
        "./data/plots/ten_vaes/paper_ready/example_approximation/grid.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    plot_example()

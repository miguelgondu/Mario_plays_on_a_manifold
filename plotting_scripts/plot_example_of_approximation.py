"""
This script plots an example approximation for Zelda.
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


def plot_example(model_id: int = 0):
    plots_dir = Path(f"./data/plots/journal_version/example_approximation_{model_id}")
    plots_dir.mkdir(exist_ok=True, parents=True)

    vae_path = Path(f"./trained_models/zelda/zelda_hierarchical_final_{model_id}.pt")
    a = np.load(f"./data/processed/zelda/grammar_checks/{vae_path.stem}.npz")
    zs = a["zs"]
    p = a["playabilities"]
    p_map = load_arrays_as_map(zs, p)
    obstacles = grid_from_map(p_map)

    ddg = DiscretizedGeometry(
        p_map,
        f"zelda_ddg_for_plotting_{model_id}",
        vae_path,
        n_grid=100,
        x_lims=(-4, 4),
        y_lims=(-4, 4),
        # force=True,
    )
    approximation = ddg.grid

    a_mv = np.load(
        f"./data/processed/metric_volumes/zelda_ddg_for_plotting_{model_id}/{vae_path.stem}.npz"
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
        plots_dir / "obstacles.png",
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
        plots_dir / "metric_volume.png",
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
        plots_dir / "approximation.png",
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
        plots_dir / "grid.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    for model_id in [3]:
        plot_example(model_id=model_id)

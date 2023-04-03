"""
This script plots all heatmaps for SMB and Zelda.
The path for the dissertation is set in stone,
so don't expect this file to run in a different
computer.
"""

from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from geometries import BaselineGeometry, DiscretizedGeometry

from vae_models.vae_mario_hierarchical import VAEMarioHierarchical
from utils.experiment import (
    load_csv_as_map,
    grid_from_map,
    load_arrays_as_map,
    load_model_from_path,
)

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
PLOTS_PATH = Path(
    "/Users/migd/Projects/dissertation/Figures/Chapter_9/all_heatmaps_and_grids"
)


def plot_grid_for_ground_truth_mario(path_to_gt: Path):
    """
    Saves the heatmap and grid from a given path.
    """
    p_map = load_csv_as_map(path_to_gt)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    folder = "ten_vaes" if "mario" in path_to_gt.stem else "zelda"

    # Plot the heatmap
    bg = BaselineGeometry(
        p_map,
        "baseline_for_plotting_dissertation",
        vae_path=ROOT_DIR / "trained_models" / folder / f"{path_to_gt.stem}.pt",
    )

    ax.imshow(bg.grid, vmin=0.0, vmax=1.0, cmap="Blues", extent=[-5, 5, -5, 5])
    ax.axis("off")

    fig.savefig(
        PLOTS_PATH / f"heatmap_{path_to_gt.stem}.jpg", dpi=120, bbox_inches="tight"
    )
    plt.close(fig)

    # Plot the grid of levels
    fig_grid, ax_grid = plt.subplots(1, 1, figsize=(7, 7))
    vae = load_model_from_path(bg.vae_path)
    vae.plot_grid(ax=ax_grid)
    ax_grid.axis("off")

    fig_grid.savefig(
        PLOTS_PATH / f"grid_{path_to_gt.stem}.jpg", dpi=120, bbox_inches="tight"
    )
    plt.close(fig_grid)


def plot_grid_and_heatmap_for_zelda(path_to_gt: Path):
    vae_path = ROOT_DIR / "trained_models" / "zelda" / f"{path_to_gt.stem}.pt"

    a = np.load(path_to_gt)
    zs = a["zs"]
    p = a["playabilities"]

    p_map = load_arrays_as_map(zs, p)
    obstacles = grid_from_map(p_map)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(obstacles, extent=[-4, 4, -4, 4], cmap="Blues", vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.savefig(
        PLOTS_PATH / f"heatmaps_{path_to_gt.stem}.jpg", dpi=120, bbox_inches="tight"
    )
    plt.close(fig)

    # Load the VAE and print a grid.
    fig_grid, ax_grid = plt.subplots(1, 1, figsize=(7, 7))
    vae = load_model_from_path(vae_path)
    vae.plot_grid(ax=ax_grid, n_rows=6, n_cols=6)
    ax_grid.axis("off")

    fig_grid.savefig(
        PLOTS_PATH / f"grid_{path_to_gt.stem}.jpg", dpi=120, bbox_inches="tight"
    )
    plt.close(fig_grid)

    # ddg = DiscretizedGeometry(
    #     p_map,
    #     f"zelda_ddg_for_plotting_{path_to_gt.stem}",
    #     vae_path,
    #     n_grid=100,
    #     x_lims=(-4, 4),
    #     y_lims=(-4, 4),
    #     # force=True,
    # )
    # approximation = ddg.grid


if __name__ == "__main__":
    all_ground_truth_paths_mario = (
        ROOT_DIR / "data" / "array_simulation_results" / "ten_vaes" / "ground_truth"
    ).glob("*.csv")

    all_ground_truth_paths_zelda = (
        ROOT_DIR / "data" / "processed" / "zelda" / "grammar_checks"
    ).glob("*.npz")

    for path_to_gt in all_ground_truth_paths_mario:
        plot_grid_for_ground_truth_mario(path_to_gt)

    for path_to_gt in all_ground_truth_paths_zelda:
        plot_grid_and_heatmap_for_zelda(path_to_gt)

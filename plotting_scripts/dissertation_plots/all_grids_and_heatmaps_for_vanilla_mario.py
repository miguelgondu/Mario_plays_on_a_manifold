"""
This script plots all heatmaps for a vanilla VAE
trained on SMB.

We use our open source version: TODO:add
"""

from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from vae_models.vae_mario_hierarchical import VAEMarioHierarchical
from vae_models.vae_vanilla_mario import VAEMario
from utils.experiment import (
    load_csv_as_map,
    load_csv_as_grid,
)

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
PLOTS_PATH = Path(
    "/Users/migd/Projects/dissertation/Figures/Chapter_9/all_heatmaps_and_grids"
)


def plot_grid_from_results(path_to_results: Path, plot_latent_codes: bool = False):
    grid = load_csv_as_grid(path_to_results)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.imshow(grid, cmap="Blues", vmin=0.0, vmax=1.0, extent=[-5, 5, -5, 5])
    ax.axis("off")

    if plot_latent_codes:
        vae = VAEMario()
        vae.load_state_dict(
            torch.load(
                ROOT_DIR
                / "trained_models"
                / "vanilla_vae"
                / f"{path_to_results.stem}.pt"
            )
        )
        vae.eval()

        encodings = vae.encode(vae.train_data).mean.detach().numpy()
        ax.scatter(
            encodings[:, 0],
            encodings[:, 1],
            c="#FB4D3D",
            edgecolors="k",
            alpha=0.75,
        )

    fig.savefig(
        PLOTS_PATH / f"heatmap_{plot_latent_codes}_{path_to_results.stem}.jpg",
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    all_ground_truth_paths = (
        ROOT_DIR / "data" / "array_simulation_results" / "vanilla_vae"
    ).glob("*.csv")

    for path_to_gt in all_ground_truth_paths:
        plot_grid_from_results(path_to_gt, plot_latent_codes=True)

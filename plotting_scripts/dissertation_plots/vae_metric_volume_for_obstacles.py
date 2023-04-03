"""
Loads up a Vanilla VAE trained on SMB, and plots the metric
volume using the extrapolate-to-1/C.
"""
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from vae_models.vae_vanilla_mario_obstacles import VAEVanillaMarioObstacles

from utils.experiment import load_csv_as_arrays, load_csv_as_grid

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
PLOTS_PATH = Path("/Users/migd/Projects/dissertation/Figures/Chapter_9/metric_volumes")
PLOTS_PATH.mkdir(exist_ok=True)

if __name__ == "__main__":
    # Load obstacles
    vae_path = (
        ROOT_DIR
        / "trained_models"
        / "vanilla_vae"
        / "16766278177816582_mariovae_final.pt"
    )
    csv_path = (
        ROOT_DIR
        / "data"
        / "array_simulation_results"
        / "vanilla_vae"
        / "16766278177816582_mariovae_final.csv"
    )
    zs, vals = load_csv_as_arrays(csv_path)
    obstacles = zs[vals < 1.0]
    grid = load_csv_as_grid(csv_path)

    vae = VAEVanillaMarioObstacles()
    vae.load_state_dict(
        torch.load("./trained_models/vanilla_vae/16766278177816582_mariovae_final.pt")
    )
    beta = -3.5
    vae.update_obstacles(
        obstacles=torch.from_numpy(obstacles).type(torch.float32), beta=beta
    )
    print(f"beta: {torch.nn.Softplus()(torch.tensor(beta))}")

    fig_play, (ax_play) = plt.subplots(1, 1, figsize=(7, 7))
    fig_grid, (ax_grid) = plt.subplots(1, 1, figsize=(7, 7))

    ax_play.imshow(grid, extent=[-5, 5, -5, 5], cmap="Blues")
    ax_play.axis("off")

    vae.plot_metric_volume(ax=ax_grid, cmap="viridis")
    ax_grid.axis("off")

    fig_grid.savefig(
        PLOTS_PATH / "vanilla_vae_obstacles_metric_volume.jpg",
        dpi=120,
        bbox_inches="tight",
    )
    fig_play.savefig(
        PLOTS_PATH / "vanilla_vae_obstacles_playability.jpg",
        dpi=120,
        bbox_inches="tight",
    )
    plt.show()

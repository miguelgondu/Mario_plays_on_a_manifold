"""
Loads up a Vanilla VAE trained on SMB, and plots the metric
volume using the extrapolate-to-1/C.
"""
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from vae_models.vae_vanilla_mario_geometric import VAEWithCenters

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
PLOTS_PATH = Path("/Users/migd/Projects/dissertation/Figures/Chapter_9/metric_volumes")
PLOTS_PATH.mkdir(exist_ok=True)

if __name__ == "__main__":
    with open(ROOT_DIR / "data" / "processed" / "playable_levels_idxs.json") as fp:
        playable_idxs = json.load(fp)

    vae = VAEWithCenters()
    vae.load_state_dict(
        torch.load("./trained_models/vanilla_vae/16766278177816582_mariovae_final.pt")
    )
    fig_grid_before, ax_grid_before = plt.subplots(1, 1, figsize=(7, 7))
    vae.plot_grid(ax=ax_grid_before, n_rows=10, n_cols=10)

    beta = -2.5
    vae.update_centers(beta=beta)
    print(f"beta: {torch.nn.Softplus()(torch.tensor(beta))}")

    fig_grid_after, ax_grid_after = plt.subplots(1, 1, figsize=(7, 7))
    vae.plot_grid(ax=ax_grid_after, n_rows=10, n_cols=10)
    ax_grid_before.axis("off")
    ax_grid_after.axis("off")

    levels = np.load("data/processed/all_levels_onehot.npz")["levels"]
    levels = torch.from_numpy(levels).type(torch.float32)
    encodings = vae.encode(levels).mean.cpu().detach().numpy()

    fig_metric, ax_metric = plt.subplots(1, 1, figsize=(7, 7))
    vae.plot_metric_volume(ax=ax_metric, cmap="viridis")

    non_playable_idxs = [i for i in range(len(encodings)) if i not in playable_idxs]

    playable_encodings = encodings[playable_idxs]
    non_playable_encodings = encodings[non_playable_idxs]
    # ax_grid.scatter(
    #     playable_encodings[:, 0],
    #     playable_encodings[:, 1],
    #     c="green",
    #     edgecolors="black",
    #     label="Playable",
    #     alpha=0.5,
    # )
    # ax_grid.scatter(
    #     non_playable_encodings[:, 0],
    #     non_playable_encodings[:, 1],
    #     c="red",
    #     edgecolors="black",
    #     label="Non-playable",
    #     alpha=0.5,
    # )
    ax_metric.scatter(
        encodings[:, 0],
        encodings[:, 1],
        c="white",
        edgecolors="black",
        alpha=0.25,
    )
    ax_metric.axis("off")
    # ax_grid.legend()

    fig_metric.savefig(
        PLOTS_PATH / "vanilla_vae_support_original.jpg", dpi=120, bbox_inches="tight"
    )
    fig_grid_before.savefig(
        PLOTS_PATH / "vanilla_vae_grid_of_levels_before.jpg",
        dpi=120,
        bbox_inches="tight",
    )
    fig_grid_after.savefig(
        PLOTS_PATH / "vanilla_vae_grid_of_levels_after.jpg",
        dpi=120,
        bbox_inches="tight",
    )
    plt.show()

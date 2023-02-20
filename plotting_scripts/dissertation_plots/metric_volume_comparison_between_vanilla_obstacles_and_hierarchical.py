"""
We plot a comparison of the metric volume comparison
between a vanilla VAE w. obstacles and a hierarchical VAE
w. obstacles. The figure also highlights how two levels
are exactly the same when extrapolating to 1/C, and are
quite different when using the hierarchical VAE.
"""
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from utils.mario.plotting import save_level_from_array

from vae_models.vae_vanilla_mario_obstacles import VAEVanillaMarioObstacles
from vae_models.vae_mario_obstacles import VAEWithObstacles

from utils.experiment import load_csv_as_arrays, load_csv_as_grid

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
PLOTS_PATH = Path(
    "/Users/migd/Projects/dissertation/Figures/Chapter_9/metric_volume_comparison_w_hierarchical"
)
PLOTS_PATH.mkdir(exist_ok=True)

if __name__ == "__main__":
    # Load one of the vanilla w. obstacles.
    vae_vanilla_path = (
        ROOT_DIR
        / "trained_models"
        / "vanilla_vae"
        / "1676628300056795_mariovae_final.pt"
    )
    csv_vanilla_path = (
        ROOT_DIR
        / "data"
        / "array_simulation_results"
        / "vanilla_vae"
        / f"{vae_vanilla_path.stem}.csv"
    )
    zs, vals = load_csv_as_arrays(csv_vanilla_path)
    obstacles = zs[vals < 1.0]
    heatmap_vanilla = load_csv_as_grid(csv_vanilla_path)

    vae_vanilla = VAEVanillaMarioObstacles()
    vae_vanilla.load_state_dict(torch.load(vae_vanilla_path))
    beta = -2.5
    vae_vanilla.update_obstacles(
        obstacles=torch.from_numpy(obstacles).type(torch.float32), beta=beta
    )
    print(f"beta: {torch.nn.Softplus()(torch.tensor(beta))}")

    # Load a hierarchical VAE w. obstacles
    vae_path = (
        ROOT_DIR / "trained_models" / "ten_vaes" / "vae_mario_hierarchical_id_7.pt"
    )

    csv_path = (
        ROOT_DIR
        / "data"
        / "array_simulation_results"
        / "ten_vaes"
        / "ground_truth"
        / f"{vae_path.stem}.csv"
    )

    zs_hierarchical, vals_hierarchical = load_csv_as_arrays(csv_path)
    obstacles_hierarchical = zs_hierarchical[vals_hierarchical < 1.0]
    heatmap_hierarchical = load_csv_as_grid(csv_path)

    vae = VAEWithObstacles()
    vae.load_state_dict(torch.load(vae_path, map_location=vae.device))
    vae.update_obstacles(
        torch.from_numpy(obstacles_hierarchical).type(torch.float32), beta=beta
    )

    # # Plotting the heatmap for Vanilla
    # fig_vanilla_heatmap, ax_vanilla_heatmap = plt.subplots(1, 1, figsize=(7, 7))
    # ax_vanilla_heatmap.imshow(heatmap_vanilla, extent=[-5, 5, -5, 5], cmap="Blues")
    # ax_vanilla_heatmap.axis("off")
    # fig_vanilla_heatmap.savefig(
    #     PLOTS_PATH / "heatmap_vanilla.jpg", dpi=120, bbox_inches="tight"
    # )

    # # Plotting the heatmap for hierarchical
    # fig_hierarchical_heatmap, ax_hierarchical_heatmap = plt.subplots(
    #     1, 1, figsize=(7, 7)
    # )
    # ax_hierarchical_heatmap.imshow(
    #     heatmap_hierarchical, extent=[-5, 5, -5, 5], cmap="Blues"
    # )
    # ax_hierarchical_heatmap.axis("off")
    # fig_hierarchical_heatmap.savefig(
    #     PLOTS_PATH / "heatmap_hierarchical.jpg", dpi=120, bbox_inches="tight"
    # )

    # # Plotting the metric volume of vanilla
    # fig_vanilla_metric_volume, ax_vanilla_metric_volume = plt.subplots(
    #     1, 1, figsize=(7, 7)
    # )
    # vae_vanilla.plot_metric_volume(ax=ax_vanilla_metric_volume, cmap="viridis")
    # ax_vanilla_metric_volume.axis("off")
    # fig_vanilla_metric_volume.savefig(
    #     PLOTS_PATH / "metric_volume_vanilla.jpg", dpi=120, bbox_inches="tight"
    # )

    # # Plotting the metric volume of hierarchical
    # fig_hierarchical_metric_volume, ax_hierarchical_metric_volume = plt.subplots(
    #     1, 1, figsize=(7, 7)
    # )
    # vae.plot_metric_volume(ax=ax_hierarchical_metric_volume, cmap="viridis")
    # ax_hierarchical_metric_volume.axis("off")
    # fig_hierarchical_metric_volume.savefig(
    #     PLOTS_PATH / "metric_volume_hierarchical.jpg", dpi=120, bbox_inches="tight"
    # )

    # plt.close()

    # Plotting all the levels in a grid for both
    full_grid_hierarchical, probs_hierarchical = vae.plot_grid(
        n_rows=10, n_cols=10, return_probs=True
    )
    full_grid_vanilla, probs_vanilla = vae_vanilla.plot_grid(
        n_rows=10, n_cols=10, return_probs=True
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(full_grid_vanilla)
    ax.axis("off")
    fig.savefig(PLOTS_PATH / "full_grid_vanilla.jpg", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(full_grid_hierarchical)
    ax.axis("off")
    fig.savefig(PLOTS_PATH / "full_grid_hierarchical.jpg", dpi=120, bbox_inches="tight")
    plt.close(fig)

    imgs_path_vanilla = PLOTS_PATH / "all_levels_vanilla"
    imgs_path_hierarchical = PLOTS_PATH / "all_levels_hierarchical"

    imgs_path_vanilla.mkdir(exist_ok=True)
    imgs_path_hierarchical.mkdir(exist_ok=True)

    for i, probs in enumerate(probs_vanilla):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.imshow(probs[:, :, 0].detach().numpy(), vmin=0.0, vmax=1.0)
        ax.axis("off")
        fig.savefig(imgs_path_vanilla / f"{i:05d}.jpg", dpi=120, bbox_inches="tight")
        plt.close(fig)

    for i, probs in enumerate(probs_hierarchical):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.imshow(probs[:, :, 0].detach().numpy(), vmin=0.0, vmax=1.0)
        ax.axis("off")
        fig.savefig(
            imgs_path_hierarchical / f"{i:05d}.jpg", dpi=120, bbox_inches="tight"
        )
        plt.close(fig)

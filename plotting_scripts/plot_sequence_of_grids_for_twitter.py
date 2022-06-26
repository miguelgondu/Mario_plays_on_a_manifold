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

if __name__ == "__main__":
    vae_path = Path("./models/ten_vaes/vae_mario_hierarchical_id_0.pt")
    path_to_gt = (
        Path("./data/array_simulation_results/ten_vaes/ground_truth")
        / f"{vae_path.stem}.csv"
    )

    vae = VAEMarioHierarchical()
    vae.load_state_dict(t.load(vae_path, map_location=vae.device))
    vae.eval()

    path_to_imgs = Path("./data/plots/animation_for_twitter/sequence_of_grids")
    path_to_imgs.mkdir(exist_ok=True)

    path_to_gt = (
        Path("./data/array_simulation_results/ten_vaes/ground_truth")
        / f"{vae_path.stem}.csv"
    )

    p_map = load_csv_as_map(path_to_gt)
    bg = BaselineGeometry(p_map, "baseline_for_plotting", vae_path)
    layer_on_top = np.where(bg.grid == 1.0, np.NaN, bg.grid)

    for n_levels in range(1, 21):
        levels_grid = vae.plot_grid(n_rows=n_levels, n_cols=n_levels)

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        plot = ax.imshow(levels_grid, extent=[-5, 5, -5, 5])
        # plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
        # ax.set_title("Decoded levels", fontsize=BIGGER_SIZE)
        ax.axis("off")
        fig.savefig(
            path_to_imgs / f"grid_{n_levels:02d}.png",
            dpi=100,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Final plot with the red layer on top
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot = ax.imshow(levels_grid, extent=[-5, 5, -5, 5])
    ax.imshow(
        layer_on_top,
        extent=[-5, 5, -5, 5],
        alpha=0.4,
        cmap="autumn",
        vmin=0.0,
        vmax=1.0,
    )
    ax.axis("off")
    fig.savefig(
        path_to_imgs / f"final_grid_with_mask.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)

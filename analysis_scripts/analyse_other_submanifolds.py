from pathlib import Path

import matplotlib.pyplot as plt

from experiment_utils import grid_from_map, load_csv_as_arrays

# Let's check the different ground truths for their other columns

columns = [
    "marioStatus",
    "timeSpentOnLevel",
    "jumpActionsPerformed",
    "lengthOfLevelPassedPhys",
    "totalActionsPerformed",
]
for id_ in [0]:
    csv_path = Path(
        f"data/array_simulation_results/ten_vaes/ground_truth/vae_mario_hierarchical_id_{id_}.csv"
    )
    fig, axes = plt.subplots(1, len(columns), figsize=(7 * len(columns), 7))

    for ax, column in zip(axes, columns):
        zs, val = load_csv_as_arrays(csv_path, column=column)
        val_map = {tuple(z.tolist()): v for z, v in zip(zs, val)}

        grid = grid_from_map(val_map)
        if column == "jumpActionsPerformed":
            fig_jump, ax_jump = plt.subplots(1, 1, figsize=(7, 7))
            plot_jump = ax_jump.imshow(
                grid, extent=[-5, 5, -5, 5], cmap="inferno", vmin=0.0, vmax=15.0
            )
            plt.colorbar(plot_jump, ax=ax_jump, fraction=0.046, pad=0.04)
            ax_jump.axis("off")
            fig_jump.savefig(
                "./data/plots/ten_vaes/paper_ready/jumping_submanifold/jumps.png",
                dpi=100,
                bbox_inches="tight",
            )

            plot = ax.imshow(
                grid, extent=[-5, 5, -5, 5], cmap="Blues", vmin=0.0, vmax=15.0
            )
        else:
            plot = ax.imshow(grid, extent=[-5, 5, -5, 5], cmap="Blues")
        plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(column)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        f"./data/plots/ten_vaes/other_submanifolds/vae_mario_hierarchical_id_{id_}.png",
        bbox_inches="tight",
    )
    # plt.show()
    plt.close()

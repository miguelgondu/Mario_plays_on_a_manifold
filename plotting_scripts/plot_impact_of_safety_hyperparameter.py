"""
This scripts plots, in the journal version, the
figure showcasing the impact of the safety hyperparameter.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from geometries import DiscretizedGeometry
from utils.experiment import grid_from_map, load_csv_as_map

PLOTS_DIR = Path("./data/plots/journal_version/safety")
PLOTS_DIR.mkdir(exist_ok=True, parents=True)


def plot_ground_truth(model_id: int = 1):
    vae_path = Path(
        f"./trained_models/ten_vaes/vae_mario_hierarchical_id_{model_id}.pt"
    )
    path_to_gt = (
        Path("./data/array_simulation_results/ten_vaes/ground_truth")
        / f"{vae_path.stem}.csv"
    )
    p_map = load_csv_as_map(path_to_gt)
    obstacles = grid_from_map(p_map)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(obstacles, cmap="Blues")
    ax.axis("off")
    fig.savefig(
        PLOTS_DIR / f"ground_truth_{model_id}.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_safety_hyperparameter(model_id: int = 1, safety_hyperparameter: float = 1.0):
    vae_path = Path(
        f"./trained_models/ten_vaes/vae_mario_hierarchical_id_{model_id}.pt"
    )
    path_to_gt = (
        Path("./data/array_simulation_results/ten_vaes/ground_truth")
        / f"{vae_path.stem}.csv"
    )
    p_map = load_csv_as_map(path_to_gt)

    dg = DiscretizedGeometry(
        p_map,
        f"geometry_for_plotting_safety_hyp_{int(safety_hyperparameter*100)}_{model_id}",
        vae_path,
        mean_scale=safety_hyperparameter,
        force=True,
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(dg.grid, cmap="Blues")
    ax.axis("off")
    fig.savefig(
        PLOTS_DIR / f"safety_{model_id}_{int(safety_hyperparameter * 100)}.png",
        dpi=120,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    model_id = 1
    plot_ground_truth(model_id=model_id)

    # for safety_hyperparameter in [1.0, 0.9, 1.1, 0.7]:
    #     plot_safety_hyperparameter(
    #         model_id=model_id, safety_hyperparameter=safety_hyperparameter
    #     )

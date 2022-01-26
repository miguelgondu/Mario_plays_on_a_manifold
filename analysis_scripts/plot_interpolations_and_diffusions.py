"""
Let's inspect whatever was saved for the different geometries.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from geometry import BaselineGeometry, DiscreteGeometry, Geometry, NormalGeometry
from experiment_utils import load_csv_as_map


def inspect_interpolations(geometry: Geometry):
    """
    Loads up and prints all lines in exp_name
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    exp_name = geometry.exp_name
    model_name = geometry.model_name
    all_arrays = Path(f"./data/arrays/ten_vaes/interpolations/{exp_name}").glob(
        f"{model_name}_*.npz"
    )
    for array_path in all_arrays:
        a = np.load(array_path)
        zs = a["zs"]
        ax.plot(zs[:, 0], zs[:, 1])

    ax.imshow(geometry.grid, extent=[-5, 5, -5, 5], cmap="Blues")
    plots_path = Path(f"./data/plots/ten_vaes/interpolations/{exp_name}")
    plots_path.mkdir(exist_ok=True, parents=True)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(plots_path / f"{model_name}.png")
    plt.close(fig)

    # plt.show()


def inspect_diffusions(geometry: Geometry):
    """
    Loads up and prints all lines in exp_name
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    exp_name = geometry.exp_name
    model_name = geometry.model_name
    all_arrays = Path(f"./data/arrays/ten_vaes/diffusions/{exp_name}").glob(
        f"{model_name}_*.npz"
    )
    for array_path in all_arrays:
        a = np.load(array_path)
        zs = a["zs"]
        ax.scatter(zs[:, 0], zs[:, 1], c="r")

    ax.imshow(geometry.grid, extent=[-5, 5, -5, 5], cmap="Blues")
    plots_path = Path(f"./data/plots/ten_vaes/diffusions/{exp_name}")
    plots_path.mkdir(exist_ok=True, parents=True)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(plots_path / f"{model_name}.png")
    plt.close(fig)


if __name__ == "__main__":
    model_paths = Path("./models/ten_vaes").glob("*.pt")
    for vae_path in model_paths:
        gt_path = Path(
            f"./data/array_simulation_results/ten_vaes/ground_truth/{vae_path.stem}.csv"
        )
        if gt_path.exists():
            gt_p_map = load_csv_as_map(gt_path)
            baseline_geometry = BaselineGeometry(gt_p_map, "baseline_gt", vae_path)
            inspect_interpolations(baseline_geometry)
            inspect_diffusions(baseline_geometry)

            normal_geometry = NormalGeometry(gt_p_map, "normal_gt", vae_path)
            inspect_interpolations(normal_geometry)
            inspect_diffusions(normal_geometry)

            discrete_geometry = DiscreteGeometry(gt_p_map, "discrete_gt", vae_path)
            inspect_interpolations(discrete_geometry)
            inspect_diffusions(discrete_geometry)

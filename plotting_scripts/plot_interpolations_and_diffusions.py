"""
Let's inspect whatever was saved for the different geometries.
"""
from pathlib import Path

import torch as t
import numpy as np
import matplotlib.pyplot as plt

from geometry import (
    BaselineGeometry,
    DiscreteGeometry,
    Geometry,
    NormalGeometry,
    ContinuousGeometry,
)
from experiment_utils import load_csv_as_map, load_trace_as_map, intersection


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
        if "discrete" in geometry.exp_name:
            ax.scatter(zs[:, 0], zs[:, 1], c="r")
            z, z_prime = zs[0], zs[-1]
            z = t.from_numpy(z).type(t.float)
            z_prime = t.from_numpy(z_prime).type(t.float)
            full_path = geometry.interpolation._full_interpolation(z, z_prime)
            ax.plot(full_path[:, 0], full_path[:, 1], "r")
        else:
            ax.plot(zs[:, 0], zs[:, 1], "r")

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


def inspect_playability_experiment():
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

            continuous_geometry = ContinuousGeometry(
                gt_p_map, "continuous_gt", vae_path
            )
            inspect_interpolations(continuous_geometry)
            inspect_diffusions(continuous_geometry)

        for m in [100, 200, 300, 400, 500]:
            al_path = Path(f"./data/evolution_traces/ten_vaes/{vae_path.stem}.npz")
            if al_path.exists():
                al_p_map = load_trace_as_map(al_path, m)
                discrete_geometry = DiscreteGeometry(
                    al_p_map, f"discrete_AL_{m}", vae_path
                )
                inspect_interpolations(discrete_geometry)
                inspect_diffusions(discrete_geometry)

        for m in [100, 200, 300, 400, 500]:
            al_path = Path(f"./data/evolution_traces/ten_vaes/{vae_path.stem}.npz")
            if al_path.exists():
                al_p_map = load_trace_as_map(al_path, m)
                continuous_geometry = ContinuousGeometry(
                    al_p_map, f"continuous_AL_{m}", vae_path
                )
                inspect_interpolations(continuous_geometry)
                inspect_diffusions(continuous_geometry)


def inspect_no_jump_experiment():
    model_paths = Path("./models/ten_vaes").glob("*.pt")
    for vae_path in model_paths:
        gt_path = Path(
            f"./data/array_simulation_results/ten_vaes/ground_truth/{vae_path.stem}.csv"
        )
        if gt_path.exists():
            # Discrete GT strict + no jump
            playable_map = load_csv_as_map(gt_path)
            strict_playability = {
                z: 1.0 if p == 1.0 else 0.0 for z, p in playable_map.items()
            }
            jump_map = load_csv_as_map(gt_path, column="jumpActionsPerformed")
            no_jump_map = {
                z: 1.0 if jumps == 0.0 else 0.0 for z, jumps in jump_map.items()
            }
            p_map = intersection(strict_playability, no_jump_map)

            dg = DiscreteGeometry(p_map, "discrete_jump_gt", vae_path)
            inspect_interpolations(dg)
            inspect_diffusions(dg)

            bg = BaselineGeometry(p_map, "baseline_jump_gt", vae_path)
            inspect_interpolations(bg)
            inspect_diffusions(bg)


def inspect_jump_experiment():
    model_paths = Path("./models/ten_vaes").glob("*.pt")
    for vae_path in model_paths:
        gt_path = Path(
            f"./data/array_simulation_results/ten_vaes/ground_truth/{vae_path.stem}.csv"
        )
        if gt_path.exists():
            # Discrete GT strict + no jump
            playable_map = load_csv_as_map(gt_path)
            strict_playability = {
                z: 1.0 if p == 1.0 else 0.0 for z, p in playable_map.items()
            }
            jump_map = load_csv_as_map(gt_path, column="jumpActionsPerformed")
            strict_jump_map = {
                z: 1.0 if jumps > 0.0 else 0.0 for z, jumps in jump_map.items()
            }
            p_map = intersection(strict_playability, strict_jump_map)

            dg = DiscreteGeometry(p_map, "discrete_force_jump_gt", vae_path)
            inspect_interpolations(dg)
            inspect_diffusions(dg)

            bg = BaselineGeometry(p_map, "baseline_force_jump_gt", vae_path)
            inspect_interpolations(bg)
            inspect_diffusions(bg)


def inspect_strict_experiment():
    model_paths = Path("./models/ten_vaes").glob("*.pt")
    for vae_path in model_paths:
        gt_path = Path(
            f"./data/array_simulation_results/ten_vaes/ground_truth/{vae_path.stem}.csv"
        )
        if gt_path.exists():
            # Discrete GT strict + no jump
            playable_map = load_csv_as_map(gt_path)
            p_map = {z: 1.0 if p == 1.0 else 0.0 for z, p in playable_map.items()}
            # jump_map = load_csv_as_map(gt_path, column="jumpActionsPerformed")
            # strict_jump_map = {
            #     z: 1.0 if jumps > 0.0 else 0.0 for z, jumps in jump_map.items()
            # }
            # p_map = intersection(strict_playability, strict_jump_map)

            dg = DiscreteGeometry(p_map, "discrete_strict_gt", vae_path)
            inspect_interpolations(dg)
            inspect_diffusions(dg)

            bg = BaselineGeometry(p_map, "baseline_strict_gt", vae_path)
            inspect_interpolations(bg)
            inspect_diffusions(bg)

            ng = NormalGeometry(p_map, "normal_strict_gt", vae_path)
            inspect_interpolations(ng)
            inspect_diffusions(ng)


if __name__ == "__main__":
    # inspect_playability_experiment()
    # inspect_no_jump_experiment()
    # inspect_jump_experiment()
    inspect_strict_experiment()

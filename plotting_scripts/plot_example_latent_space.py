"""
Plots figure [ref], with an example of latent space,
grid of levels, interpolations and diffusions.
"""

from pathlib import Path

from utils.experiment import load_csv_as_map, load_experiment

import torch as t
import matplotlib.pyplot as plt

from geometries import DiscreteGeometry
from vae_mario_hierarchical import VAEMarioHierarchical


def plot_ground_truth(grid):
    """Plots Fig. [ref](a)"""
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(grid, cmap="Blues", extent=[-5, 5, -5, 5])
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(
        "./data/plots/ten_vaes/paper_ready/ground_truth.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_grid_of_levels(model_path):
    """Plots Fig. [ref](b)"""
    vae = VAEMarioHierarchical()
    vae.load_state_dict(t.load(model_path, map_location=vae.device))

    img = vae.plot_grid()
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(img, extent=[-5, 5, -5, 5])
    ax.axis("off")
    fig.savefig(
        "./data/plots/ten_vaes/paper_ready/grid_of_levels.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)


def load_interpolations(id_=7):
    interp_path = Path("./data/arrays/ten_vaes/interpolations")
    res_interp_path = Path("./data/array_simulation_results/ten_vaes/interpolations")

    c_interp_array = (
        interp_path / f"continuous_gt/vae_mario_hierarchical_id_0_interp_{id_:02d}.npz"
    )
    d_interp_array = (
        interp_path / f"discrete_gt/vae_mario_hierarchical_id_0_interp_{id_:02d}.npz"
    )
    b_interp_array = (
        interp_path / f"baseline_gt/vae_mario_hierarchical_id_0_interp_{id_:02d}.npz"
    )

    c_interp_csv = (
        res_interp_path
        / f"continuous_gt/vae_mario_hierarchical_id_0_interp_{id_:02d}.csv"
    )
    d_interp_csv = (
        res_interp_path
        / f"discrete_gt/vae_mario_hierarchical_id_0_interp_{id_:02d}.csv"
    )
    b_interp_csv = (
        res_interp_path
        / f"baseline_gt/vae_mario_hierarchical_id_0_interp_{id_:02d}.csv"
    )

    zs_interp_b, p_interp_b, levels_interp_b = load_experiment(
        b_interp_array, b_interp_csv
    )
    zs_interp_c, p_interp_c, levels_interp_c = load_experiment(
        c_interp_array, c_interp_csv
    )
    zs_interp_d, p_interp_d, levels_interp_d = load_experiment(
        d_interp_array, d_interp_csv
    )

    dict_ = {
        "baseline": (zs_interp_b, levels_interp_b, p_interp_b),
        "continuous": (zs_interp_c, levels_interp_c, p_interp_c),
        "discrete": (zs_interp_d, levels_interp_d, p_interp_d),
    }

    return dict_


def load_diffusions():
    diff_path = Path("./data/arrays/ten_vaes/diffusion")
    res_diff_path = Path("./data/array_simulation_results/ten_vaes/interpolations")

    b_diff_array = diff_path / "baseline_gt/vae_mario_hierarchical_id_0_diff_00.npz"
    c_diff_array = diff_path / "continuous_gt/vae_mario_hierarchical_id_0_diff_00.npz"
    d_diff_array = diff_path / "discrete_gt/vae_mario_hierarchical_id_0_diff_00.npz"
    b_diff_csv = res_diff_path / "baseline_gt/vae_mario_hierarchical_id_0_diff_00.csv"
    c_diff_csv = res_diff_path / "continuous_gt/vae_mario_hierarchical_id_0_diff_00.csv"
    d_diff_csv = res_diff_path / "discrete_gt/vae_mario_hierarchical_id_0_diff_00.csv"

    zs_diff_b, levels_diff_b, p_diff_b = load_experiment(b_diff_array, b_diff_csv)
    zs_diff_c, levels_diff_c, p_diff_c = load_experiment(c_diff_array, c_diff_csv)
    zs_diff_d, levels_diff_d, p_diff_d = load_experiment(d_diff_array, d_diff_csv)

    dict_ = {
        "baseline": (zs_diff_b, levels_diff_b, p_diff_b),
        "continuous": (zs_diff_c, levels_diff_c, p_diff_c),
        "discrete": (zs_diff_d, levels_diff_d, p_diff_d),
    }

    return dict_


def plot_example_interpolations(grid):
    # TODO: I should implement a util function that loads
    # results in a sorted way according to the original .npz
    # This is already almost done in plot_banner

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(grid, extent=[-5, 5, -5, 5], cmap="Blues")
    for id_ in [7]:
        interpolations = load_interpolations(id_=id_)
        zs_interp_b, _, p_interp_b = interpolations["baseline"]
        zs_interp_c, _, p_interp_c = interpolations["continuous"]
        zs_interp_d, _, p_interp_d = interpolations["discrete"]

        ax.plot(zs_interp_b[:, 0], zs_interp_b[:, 1], "-k", label="Linear")
        ax.scatter(
            zs_interp_b[:, 0],
            zs_interp_b[:, 1],
            c=p_interp_b,
            edgecolors="orange",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
        )

        ax.plot(zs_interp_c[:, 0], zs_interp_c[:, 1], "-r", label="Geodesic")
        ax.scatter(
            zs_interp_c[:, 0],
            zs_interp_c[:, 1],
            c=p_interp_c,
            edgecolors="orange",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
        )

        ax.plot(zs_interp_d[:, 0], zs_interp_d[:, 1], "-b", label="A star")
        ax.scatter(
            zs_interp_d[:, 0],
            zs_interp_d[:, 1],
            c=p_interp_d,
            edgecolors="orange",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
        )

    ax.legend()
    ax.axis("off")
    fig.savefig(
        "./data/plots/ten_vaes/paper_ready/examples_interpolations.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_example_diffusions(grid):
    diffusions = load_diffusions()
    zs_diff_b, _, p_diff_b = diffusions["baseline"]
    zs_diff_c, _, p_diff_c = diffusions["continuous"]
    zs_diff_d, _, p_diff_d = diffusions["discrete"]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.imshow(grid, extent=[-5, 5, -5, 5], cmap="Blues")
    ax.axis("off")
    fig.savefig(
        "./data/plots/ten_vaes/paper_ready/grid_of_levels.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    # model_name = "vae_mario_hierarchical_id_0"
    # gt_csv = Path(
    #     f"./data/array_simulation_results/ten_vaes/ground_truth/{model_name}.csv"
    # )
    # model_path = Path("./models/ten_vaes") / f"{model_name}.pt"
    # p_map = load_csv_as_map(gt_csv)
    # geometry = DiscreteGeometry(p_map, "discrete_plotting", model_path)

    # plot_ground_truth(geometry.grid)
    # plot_grid_of_levels(model_path)
    # plot_example_interpolations(geometry.grid)
    for model_path in Path("./models/MarioGAN").glob("*.pth"):
        model_name = model_path.stem
        gt_csv = Path(
            f"./data/array_simulation_results/MarioGAN/ground_truth/{model_name}.csv"
        )
        p_map = load_csv_as_map(gt_csv)
        geometry = DiscreteGeometry(p_map, "discrete_plotting_mariogan", model_path)
        # plot_ground_truth(geometry.grid)
        _, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.imshow(geometry.grid, extent=[-10, 10, -10, 10], cmap="Blues")
        ax.set_title(model_name)
        plt.show()

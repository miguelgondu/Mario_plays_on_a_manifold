"""
This scripts showcases the main point of the paper
by doing 3 interpolations, where the linear baseline
is sure to perform poorly in comparison to our geometries.
"""
from pathlib import Path

import torch as t
import numpy as np
import matplotlib.pyplot as plt

from simulate_array import _simulate_array
from geometry import DiscreteGeometry, ContinuousGeometry, BaselineGeometry
from experiment_utils import load_csv_as_map, build_discretized_manifold
from mario_utils.plotting import get_img_from_level


def saving_the_interpolations_as_arrays():
    """
    Interpolate between [] and [] using
    the three methods
    """
    z = t.Tensor([0.0, -4.0])
    z_prime = t.Tensor([3.5, -4.0])
    (
        d_interp,
        d_levels,
        b_interp,
        b_levels,
        c_interp,
        c_levels,
        _,
    ) = get_interpolations_and_levels(z, z_prime)

    np.savez(
        "./data/arrays/ten_vaes/final_plots/banner_plot_discrete.npz",
        zs=d_interp.detach().numpy(),
        levels=d_levels.detach().numpy(),
    )
    np.savez(
        "./data/arrays/ten_vaes/final_plots/banner_plot_continuous.npz",
        zs=c_interp.detach().numpy(),
        levels=c_levels.detach().numpy(),
    )
    np.savez(
        "./data/arrays/ten_vaes/final_plots/banner_plot_baseline.npz",
        zs=b_interp.detach().numpy(),
        levels=b_levels.detach().numpy(),
    )

    # plot_interpolations(d_interp, c_interp, b_interp, grid)
    # plot_levels(d_levels, "discrete")
    # plot_levels(b_levels, "baseline")
    # plot_levels(c_levels, "continuous")


def get_interpolations_and_levels(z, z_prime):
    vae_path = Path("./models/ten_vaes/vae_mario_hierarchical_id_0.pt")
    model_name = vae_path.stem
    path_to_gt = (
        Path("./data/array_simulation_results/ten_vaes/ground_truth")
        / f"{model_name}.csv"
    )
    p_map = load_csv_as_map(path_to_gt)

    dg = DiscreteGeometry(p_map, "discrete_gt", vae_path)
    bg = BaselineGeometry(p_map, "baseline_gt", vae_path)

    print(f"Building the discretized manifold for {vae_path}")
    manifold = build_discretized_manifold(p_map, vae_path)
    cg = ContinuousGeometry(p_map, "continuous_gt", vae_path, manifold=manifold)

    z = t.from_numpy(dg.interpolation._query_tree(z.detach().numpy())).type(t.float)
    z_prime = t.from_numpy(dg.interpolation._query_tree(z_prime.detach().numpy())).type(
        t.float
    )

    d_interp, d_levels = dg.interpolate(z, z_prime)
    b_interp, b_levels = bg.interpolate(z, z_prime)
    c_interp, c_levels = cg.interpolate(z, z_prime)

    return d_interp, d_levels, b_interp, b_levels, c_interp, c_levels, bg.grid


def plot_levels(levels, name):
    images = [get_img_from_level(lvl.detach().numpy()) for lvl in levels]
    final_img = np.concatenate(images, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.imshow(final_img)
    ax.axis("off")
    fig.savefig(f"./data/plots/ten_vaes/paper_ready/levels_{name}.png", dpi=100)
    plt.show()
    plt.close()


def plot_interpolations(d_interp, c_interp, b_interp, grid):
    d_interp = d_interp.detach().numpy()
    c_interp = c_interp.detach().numpy()
    b_interp = b_interp.detach().numpy()

    _, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.plot(d_interp[:, 0], d_interp[:, 1], "b--", linewidth=5)
    ax.plot(c_interp[:, 0], c_interp[:, 1], "r", linewidth=5)
    ax.plot(b_interp[:, 0], b_interp[:, 1], "k:", linewidth=5)
    ax.scatter(d_interp[:, 0], d_interp[:, 1], c="k", s=50, zorder=4, edgecolors="w")
    ax.scatter(c_interp[:, 0], c_interp[:, 1], c="k", s=50, zorder=4, edgecolors="w")
    ax.scatter(b_interp[:, 0], b_interp[:, 1], c="k", s=50, zorder=4, edgecolors="w")

    ax.imshow(grid, extent=[-5, 5, -5, 5], cmap="Blues")
    ax.set_xlim([-0.2, 4.3])
    ax.set_ylim([-4.2, -1.8])

    plt.tight_layout()
    plt.savefig("./data/plots/ten_vaes/paper_ready/example_interpolations.png", dpi=100)
    plt.show()
    plt.close()

    # Getting the levels as figures


def loading_results():
    """
    Gets the csvs and returns points and levels
    """
    pass


def plot():
    """
    Plots figure [ref].
    """
    pass


if __name__ == "__main__":
    # saving_the_interpolations_as_arrays()
    # _simulate_array(
    #     "./data/arrays/ten_vaes/final_plots/banner_plot_baseline.npz",
    #     10,
    #     5,
    #     "ten_vaes/final_plots",
    # )
    # _simulate_array(
    #     "./data/arrays/ten_vaes/final_plots/banner_plot_continuous.npz",
    #     10,
    #     5,
    #     "ten_vaes/final_plots",
    # )
    # _simulate_array(
    #     "./data/arrays/ten_vaes/final_plots/banner_plot_discrete.npz",
    #     10,
    #     5,
    #     "ten_vaes/final_plots",
    # )
    pass
    # TODO: implement data loading.

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
from utils.experiment import (
    load_csv_as_arrays,
    load_csv_as_map,
    build_discretized_manifold,
)
from utils.mario.plotting import get_img_from_level


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


def paint_red_x(level, p):
    # TODO implement
    if p == 0:
        h, w, _ = level.shape
        red = [255, 0, 0]
        for i, j in zip(range(h // 5, (4 * h) // 5), range(w // 5, (4 * w) // 5)):
            level[i, j, :] = red
            level[i + np.arange(15), j, :] = red
            level[i, j + np.arange(15), :] = red
        return level
    else:
        return level


def plot_levels(levels, p, name):
    images = [paint_red_x(get_img_from_level(lvl), p_) for lvl, p_ in zip(levels, p)]
    final_img = np.concatenate(images, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 1))
    ax.imshow(final_img)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(
        f"./data/plots/ten_vaes/paper_ready/levels_{name}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def plot_interpolations(d_interp, p_d, c_interp, p_c, b_interp, p_b, grid):
    # d_interp = d_interp.detach().numpy()
    # c_interp = c_interp.detach().numpy()
    # b_interp = b_interp.detach().numpy()

    _, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.plot(d_interp[:, 0], d_interp[:, 1], "b--", linewidth=5)
    ax.plot(c_interp[:, 0], c_interp[:, 1], "r", linewidth=5)
    ax.plot(b_interp[:, 0], b_interp[:, 1], "k:", linewidth=5)
    ax.scatter(
        d_interp[:, 0],
        d_interp[:, 1],
        c=p_d,
        s=50,
        zorder=4,
        edgecolors="orange",
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
    )
    ax.scatter(
        c_interp[:, 0],
        c_interp[:, 1],
        c=p_c,
        s=50,
        zorder=4,
        edgecolors="orange",
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
    )
    ax.scatter(
        b_interp[:, 0],
        b_interp[:, 1],
        c=p_b,
        s=50,
        zorder=4,
        edgecolors="orange",
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
    )

    ax.imshow(grid, extent=[-5, 5, -5, 5], cmap="Blues")
    ax.set_xlim([-0.2, 4.3])
    ax.set_ylim([-4.2, -1.8])

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        "./data/plots/ten_vaes/paper_ready/example_interpolations_after_sim.png",
        dpi=100,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def loading_results():
    """
    Gets the csvs and returns points, results and levels
    """
    res_path = Path("./data/array_simulation_results/ten_vaes/final_plots")
    array_path = Path("./data/arrays/ten_vaes/final_plots")
    zs_b, p_b = load_csv_as_arrays(res_path / f"banner_plot_baseline.csv")
    zs_c, p_c = load_csv_as_arrays(res_path / f"banner_plot_continuous.csv")
    zs_d, p_d = load_csv_as_arrays(res_path / f"banner_plot_discrete.csv")

    zs_b_prime = np.load(array_path / f"banner_plot_baseline.npz")["zs"]
    levels_b = np.load(array_path / f"banner_plot_baseline.npz")["levels"]
    zs_c_prime = np.load(array_path / f"banner_plot_continuous.npz")["zs"]
    levels_c = np.load(array_path / f"banner_plot_continuous.npz")["levels"]
    zs_d_prime = np.load(array_path / f"banner_plot_discrete.npz")["zs"]
    levels_d = np.load(array_path / f"banner_plot_discrete.npz")["levels"]

    # Instead, I should just reorder the ps according to zs_c_prime
    p_b = [p_b[zs_b.tolist().index(z.tolist())] for z in zs_b_prime]
    p_c = [p_c[zs_c.tolist().index(z.tolist())] for z in zs_c_prime]
    p_d = [p_d[zs_d.tolist().index(z.tolist())] for z in zs_d_prime]

    return (
        zs_b_prime,
        levels_b,
        p_b,
        zs_c_prime,
        levels_c,
        p_c,
        zs_d_prime,
        levels_d,
        p_d,
    )


def plot():
    """
    Plots figure [ref].
    """
    # Getting the interpolations and levels
    (
        zs_b_prime,
        levels_b,
        p_b,
        zs_c_prime,
        levels_c,
        p_c,
        zs_d_prime,
        levels_d,
        p_d,
    ) = loading_results()

    # Getting the grid itself
    vae_path = Path("./models/ten_vaes/vae_mario_hierarchical_id_0.pt")
    model_name = vae_path.stem
    path_to_gt = (
        Path("./data/array_simulation_results/ten_vaes/ground_truth")
        / f"{model_name}.csv"
    )
    p_map = load_csv_as_map(path_to_gt)
    dg = DiscreteGeometry(p_map, "discrete_gt", vae_path)
    grid = dg.grid

    # Plotting
    plot_interpolations(zs_d_prime, p_d, zs_c_prime, p_c, zs_b_prime, p_b, grid)
    plot_levels(levels_b, p_b, "baseline_after_sim")
    plot_levels(levels_c, p_c, "continuous_after_sim")
    plot_levels(levels_d, p_d, "discrete_after_sim")


def simulate_arrays():
    _simulate_array(
        "./data/arrays/ten_vaes/final_plots/banner_plot_baseline.npz",
        10,
        5,
        "ten_vaes/final_plots",
    )
    _simulate_array(
        "./data/arrays/ten_vaes/final_plots/banner_plot_continuous.npz",
        10,
        5,
        "ten_vaes/final_plots",
    )
    _simulate_array(
        "./data/arrays/ten_vaes/final_plots/banner_plot_discrete.npz",
        10,
        5,
        "ten_vaes/final_plots",
    )


if __name__ == "__main__":
    # saving_the_interpolations_as_arrays()
    # simulate_arrays()
    plot()

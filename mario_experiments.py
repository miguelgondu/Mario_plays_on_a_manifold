from typing import Tuple, List
import json

import torch as t
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from vae_mario_hierarchical import VAEMarioHierarchical
from vae_geometry_hierarchical import VAEGeometryHierarchical

from geoml.discretized_manifold import DiscretizedManifold

from interpolations.linear_interpolation import LinearInterpolation
from interpolations.geodesic_interpolation import GeodesicInterpolation
from diffusions.normal_diffusion import NormalDifussion
from diffusions.baseline_diffusion import BaselineDiffusion
from diffusions.geometric_difussion import GeometricDifussion

from metric_approximation_with_jacobians import approximate_metric, plot_approximation
from toy_experiment import get_random_pairs, get_interpolations


def figure_grid_levels(vae: VAEGeometryHierarchical):
    """
    Plots a grid of levels
    """
    _, ax = plt.subplots(1, 1, figsize=(7, 7))
    vae.plot_grid(ax=ax, sample=False)

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        "./data/plots/final/mario_grid_of_levels.png", dpi=100, bbox_inches="tight"
    )
    plt.show()
    plt.close()


def figure_metric_for_beta(
    vae: VAEGeometryHierarchical, beta: float = -3.0, n_clusters: int = 50
):
    vae.update_cluster_centers(beta=beta, n_clusters=n_clusters)

    _, ax = plt.subplots(1, 1, figsize=(7, 7))
    vae.plot_metric_volume(ax=ax)

    zs = vae.encodings.detach().numpy()
    ax.scatter(zs[:, 0], zs[:, 1], marker="x", c="#DC851F")

    ax.set_title(r"$\beta=$" + f"{beta}", fontsize=20)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        f"./data/plots/final/mario_metric_beta_{beta}.png", dpi=100, bbox_inches="tight"
    )
    plt.show()
    plt.close()


def figure_metric_for_different_betas(
    vae: VAEGeometryHierarchical, n_clusters: int = 50
):
    for beta in [-2.0, -2.5, -3.0, -3.5]:
        figure_metric_for_beta(vae, beta=beta, n_clusters=n_clusters)

    # Also plotting one for vae.decoder.
    _, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot_approximation(vaeh, function=vaeh.decode, ax=ax)
    zs = vae.encodings.detach().numpy()
    ax.scatter(zs[:, 0], zs[:, 1], marker="x", c="#DC851F")
    ax.set_title("No UQ", fontsize=20)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        f"./data/plots/final/mario_metric_no_UQ.png", dpi=100, bbox_inches="tight"
    )
    plt.show()
    plt.close()


def get_ground_truth() -> np.ndarray:
    df = pd.read_csv(
        "./data/processed/ground_truth/hierarchical_final_playable_final_ground_truth.csv"
    )
    playability = df.groupby(["z1", "z2"])["marioStatus"].mean()
    z1 = np.array(list(set([idx[0] for idx in playability.index.values])))
    z1 = np.sort(z1)
    z2 = np.array(list(set([idx[1] for idx in playability.index.values])))
    z2 = np.sort(z2)

    positions = {
        (x.item(), y.item()): (i, j)
        for j, x in enumerate(z1)
        for i, y in enumerate(reversed(z2))
    }

    playability_img = np.zeros((len(z1), len(z2)))
    for z, (i, j) in positions.items():
        (x, y) = z
        p = playability[(x, y)]
        playability_img[i, j] = p

    return playability_img


def ground_truth_plot(vae):
    """
    Grabs the ground truth table and computes the average
    playability of each position. Compares with the latent space.
    """

    playability_img = get_ground_truth()
    _, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.imshow(playability_img, extent=[-5, 5, -5, 5], cmap="Blues")
    ax.axis("off")

    zs = vae.encodings.detach().numpy()
    ax.scatter(zs[:, 0], zs[:, 1], marker="x", c="#DC851F")
    plt.tight_layout()
    plt.savefig(
        f"./data/plots/final/mario_ground_truth.png", dpi=100, bbox_inches="tight"
    )
    plt.show()
    plt.close()


def save_interpolations(vae: VAEGeometryHierarchical, seed: int = 0):
    """
    Saves an array for further simulation.
    """
    z_0s, z_1s = get_random_pairs(vae.encodings, n_pairs=50, seed=seed)
    li, gi = get_interpolations(vae)

    fifty_lines = [li.interpolate(z_0, z_1) for z_0, z_1 in zip(z_0s, z_1s)]

    fifty_geodesics_splines = [
        gi.interpolate_and_return_geodesic(z_0, z_1) for z_0, z_1 in zip(z_0s, z_1s)
    ]
    domain = t.linspace(0, 1, gi.n_points_in_line)
    fifty_geodesics = [c(domain) for c in fifty_geodesics_splines]

    # Decode lines and geodesics into levels, and save the arrays.
    fifty_lines = t.cat(fifty_lines)
    fifty_geodesics = t.cat(fifty_geodesics)
    fifty_lines_levels = vae.decode(fifty_lines).probs.argmax(dim=-1)
    fifty_geodesics_levels = vae.decode(fifty_geodesics).probs.argmax(dim=-1)

    print(fifty_lines_levels[0])
    print(fifty_geodesics_levels[0])

    print("Saving arrays")
    np.savez(
        "./data/arrays/fifty_lines_and_levels.npz",
        zs=fifty_lines.detach().numpy(),
        levels=fifty_lines_levels.detach().numpy(),
    )
    np.savez(
        "./data/arrays/fifty_geodesics_and_levels.npz",
        zs=fifty_geodesics.detach().numpy(),
        levels=fifty_geodesics_levels.detach().numpy(),
    )


def compare_ground_truth_and_metric_volume(vae: VAEGeometryHierarchical):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(7 * 2, 7))
    vae.plot_metric_volume(ax=ax1)
    ground_truth = get_ground_truth()
    plot = ax2.imshow(
        ground_truth, extent=[-5, 5, -5, 5], vmin=0.0, vmax=1.0, cmap="Blues"
    )
    plt.colorbar(plot, ax=ax2, fraction=0.046, pad=0.04)

    ax1.set_title("Metric volume")
    ax2.set_title("Ground truth of playability")

    plt.show()


def analyse_interpolation_results(vae: VAEGeometryHierarchical) -> List[np.ndarray]:
    """
    Loads up the experiment csvs after simulation
    """
    df_lines = pd.read_csv(
        "./data/array_simulation_results/fifty_lines_and_levels.csv", index_col=0
    )
    df_geodesics = pd.read_csv(
        "./data/array_simulation_results/fifty_geodesics_and_levels.csv", index_col=0
    )

    groupby_geodesics = df_geodesics.groupby("z")
    playability_geodesics = groupby_geodesics["marioStatus"].mean()
    groupby_lines = df_lines.groupby("z")
    playability_lines = groupby_lines["marioStatus"].mean()
    print(playability_geodesics)
    print(playability_lines)

    print(f"Mean playability for linear: {np.mean(playability_lines)}")
    print(f"Mean playability for geodesic: {np.mean(playability_geodesics)}")

    zs_lines = np.array([json.loads(z) for z in playability_lines.index])
    zs_geodesics = np.array([json.loads(z) for z in playability_geodesics.index])
    print(zs_lines)
    print(zs_geodesics)

    c_lines = [playability_lines.loc[z] for z in playability_lines.index]
    c_geodesics = [playability_lines.loc[z] for z in playability_lines.index]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(7 * 2, 7))
    vae.plot_metric_volume(ax=ax1)
    ground_truth = get_ground_truth()
    ax2.imshow(ground_truth, extent=[-5, 5, -5, 5])
    ax1.scatter(zs_lines[:, 0], zs_lines[:, 1], c=c_lines, vmin=0.0, vmax=1.0)
    ax1.scatter(
        zs_geodesics[:, 0], zs_geodesics[:, 1], c=c_geodesics, vmin=0.0, vmax=1.0
    )

    # There are 10 zs per line/geodesic.
    # so we need to do some hacking to plot the lines/geodesics.
    # for k in range(50):
    #     ax1.plot(
    #         zs_lines[k * 10 : (k + 1) * 10 + 1, 0],
    #         zs_lines[k * 10 : (k + 1) * 10 + 1, 1],
    #         "--r",
    #     )
    #     ax1.plot(
    #         zs_geodesics[k * 10 : (k + 1) * 10 + 1, 0],
    #         zs_geodesics[k * 10 : (k + 1) * 10 + 1, 1],
    #         "--r",
    #     )
    #     ax2.plot(
    #         zs_lines[k * 10 : (k + 1) * 10 + 1, 0],
    #         zs_lines[k * 10 : (k + 1) * 10 + 1, 1],
    #         "--r",
    #     )
    #     ax2.plot(
    #         zs_geodesics[k * 10 : (k + 1) * 10 + 1, 0],
    #         zs_geodesics[k * 10 : (k + 1) * 10 + 1, 1],
    #         "--r",
    #     )

    plt.show()

    return playability_lines, playability_geodesics


def get_diffusions() -> Tuple[NormalDifussion, BaselineDiffusion, GeometricDifussion]:
    n_points = 50

    normal_diffusion = NormalDifussion(n_points, scale=0.5)
    geometric_diffusion = GeometricDifussion(n_points, scale=0.5)
    baseline_diffusion = BaselineDiffusion(n_points, step_size=0.5)

    return normal_diffusion, baseline_diffusion, geometric_diffusion


def save_diffusion_experiment(vae):
    """
    Saves arrays for the diffusion experiment.
    """
    n_runs = 10
    normal_diff, baseline_diff, geometric_diff = get_diffusions()

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)

    # vae.plot_latent_space(ax=(ax1, ax2, ax3))

    all_zs_n = []
    all_zs_b = []
    all_zs_g = []
    for _ in range(n_runs):
        zs_n = normal_diff.run(vae)
        zs_b = baseline_diff.run(vae)
        zs_g = geometric_diff.run(vae)

        all_zs_n.append(zs_n)
        all_zs_b.append(zs_b)
        all_zs_g.append(zs_g)

        zs_n = zs_n.detach().numpy()
        zs_b = zs_b.detach().numpy()
        zs_g = zs_g.detach().numpy()

        ax1.scatter(zs_n[:, 0], zs_n[:, 1], c="c")
        ax2.scatter(zs_b[:, 0], zs_b[:, 1], c="r")
        ax3.scatter(zs_g[:, 0], zs_g[:, 1], c="g")

    ground_truth = get_ground_truth()
    for ax in [ax1, ax2, ax3]:
        plot = ax.imshow(
            ground_truth, extent=[-5, 5, -5, 5], vmin=0.0, vmax=1.0, cmap="Blues"
        )
        plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlim((-5, 5))
        ax.axis("off")

    ax1.set_title("Normal")
    ax2.set_title("Baseline")
    ax3.set_title("Geometric")
    plt.tight_layout()
    plt.show()

    all_zs_n = t.vstack(all_zs_n)
    all_zs_b = t.vstack(all_zs_b)
    all_zs_g = t.vstack(all_zs_g)

    levels_n = vae.decode(all_zs_n).probs.argmax(dim=-1)
    levels_b = vae.decode(all_zs_b).probs.argmax(dim=-1)
    levels_g = vae.decode(all_zs_g).probs.argmax(dim=-1)

    np.savez(
        "./data/arrays/normal_diffusion.npz",
        zs=all_zs_n.detach().numpy(),
        levels=levels_n.detach().numpy(),
    )
    np.savez(
        "./data/arrays/baseline_diffusion.npz",
        zs=all_zs_b.detach().numpy(),
        levels=levels_b.detach().numpy(),
    )
    np.savez(
        "./data/arrays/geometric_diffusion.npz",
        zs=all_zs_g.detach().numpy(),
        levels=levels_g.detach().numpy(),
    )


def plot_saved_diffusions(vae: VAEGeometryHierarchical):
    zs_n = np.load("./data/arrays/normal_diffusion.npz")["zs"]
    zs_b = np.load("./data/arrays/baseline_diffusion.npz")["zs"]
    zs_g = np.load("./data/arrays/geometric_diffusion.npz")["zs"]
    levels_n = np.load("./data/arrays/normal_diffusion.npz")["levels"]
    levels_b = np.load("./data/arrays/baseline_diffusion.npz")["levels"]
    levels_g = np.load("./data/arrays/geometric_diffusion.npz")["levels"]

    print(len(np.unique(levels_n, axis=0)))
    print(len(np.unique(levels_b, axis=0)))
    print(len(np.unique(levels_g, axis=0)))

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
    ax1.scatter(zs_n[:, 0], zs_n[:, 1], c="c")
    ax2.scatter(zs_b[:, 0], zs_b[:, 1], c="r")
    ax3.scatter(zs_g[:, 0], zs_g[:, 1], c="g")

    ground_truth = get_ground_truth()
    for ax in [ax1, ax2, ax3]:
        plot = ax.imshow(
            ground_truth, extent=[-5, 5, -5, 5], vmin=0.0, vmax=1.0, cmap="Blues"
        )
        plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlim((-5, 5))
        ax.axis("off")

    ax1.set_title("Normal random walk")
    ax2.set_title("Baseline")
    ax3.set_title("Geometric random walk")
    plt.tight_layout()
    plt.show()


def analyse_diffusion_experiment(vae: VAEGeometryHierarchical):
    df_normal = pd.read_csv(
        "./data/array_simulation_results/normal_diffusion.csv", index_col=0
    )
    df_baseline = pd.read_csv(
        "./data/array_simulation_results/baseline_diffusion.csv", index_col=0
    )
    df_geometric = pd.read_csv(
        "./data/array_simulation_results/geometric_diffusion.csv", index_col=0
    )

    playability_normal = df_normal.groupby("z")["marioStatus"].mean()
    playability_baseline = df_baseline.groupby("z")["marioStatus"].mean()
    playability_geometric = df_geometric.groupby("z")["marioStatus"].mean()

    print(playability_normal)
    print(playability_baseline)
    print(playability_geometric)
    print(f"Mean playability for normal: {np.mean(playability_normal)}")
    print(f"Mean playability for baseline: {np.mean(playability_baseline)}")
    print(f"Mean playability for geodesic: {np.mean(playability_geometric)}")


def run_diffusion_on_ground_truth(vae: VAEGeometryHierarchical):
    df = pd.read_csv(
        "./data/processed/ground_truth/hierarchical_final_playable_final_ground_truth.csv"
    )
    playability = df.groupby(["z1", "z2"])["marioStatus"].mean()
    good_zs = [z for z, v in playability.iteritems() if v == 1.0]
    print(good_zs)
    print(np.array(good_zs))

    vae.update_cluster_centers(
        beta=vae.translated_sigmoid.beta, cluster_centers=t.Tensor(good_zs)
    )

    z_0s = [
        vae.cluster_centers[np.random.randint(len(vae.cluster_centers))]
        for _ in range(10)
    ]
    normal_diff, baseline_diff, geometric_diff = get_diffusions()

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)

    # vae.plot_latent_space(ax=(ax1, ax2, ax3))

    all_zs_n = []
    all_zs_b = []
    all_zs_g = []
    for z_0 in z_0s:
        zs_n = normal_diff.run(vae, z_0=z_0)
        zs_b = baseline_diff.run(vae, z_0=z_0)
        zs_g = geometric_diff.run(vae, z_0=z_0)

        all_zs_n.append(zs_n)
        all_zs_b.append(zs_b)
        all_zs_g.append(zs_g)

        zs_n = zs_n.detach().numpy()
        zs_b = zs_b.detach().numpy()
        zs_g = zs_g.detach().numpy()

        ax1.scatter(zs_n[:, 0], zs_n[:, 1], c="#DC851F")
        ax2.scatter(zs_b[:, 0], zs_b[:, 1], c="#DC851F")
        ax3.scatter(zs_g[:, 0], zs_g[:, 1], c="#DC851F")

    ground_truth = get_ground_truth()
    for ax in [ax1, ax2, ax3]:
        plot = ax.imshow(
            ground_truth, extent=[-5, 5, -5, 5], vmin=0.0, vmax=1.0, cmap="Blues"
        )
        plt.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlim((-5, 5))
        ax.axis("off")

    ax1.set_title("Normal")
    ax2.set_title("Baseline")
    ax3.set_title("Geometric")
    plt.tight_layout()
    plt.show()

    all_zs_n = t.vstack(all_zs_n)
    all_zs_b = t.vstack(all_zs_b)
    all_zs_g = t.vstack(all_zs_g)

    levels_n = vae.decode(all_zs_n).probs.argmax(dim=-1)
    levels_b = vae.decode(all_zs_b).probs.argmax(dim=-1)
    levels_g = vae.decode(all_zs_g).probs.argmax(dim=-1)

    np.savez(
        "./data/arrays/normal_diffusion_ground_truth.npz",
        zs=all_zs_n.detach().numpy(),
        levels=levels_n.detach().numpy(),
    )
    np.savez(
        "./data/arrays/baseline_diffusion_ground_truth.npz",
        zs=all_zs_b.detach().numpy(),
        levels=levels_b.detach().numpy(),
    )
    np.savez(
        "./data/arrays/geometric_diffusion_ground_truth.npz",
        zs=all_zs_g.detach().numpy(),
        levels=levels_g.detach().numpy(),
    )


def analyse_diffusion_experiment_on_ground_truth(vae: VAEGeometryHierarchical):
    df_normal = pd.read_csv(
        "./data/array_simulation_results/normal_diffusion_ground_truth.csv", index_col=0
    )
    df_baseline = pd.read_csv(
        "./data/array_simulation_results/baseline_diffusion_ground_truth.csv",
        index_col=0,
    )
    df_geometric = pd.read_csv(
        "./data/array_simulation_results/geometric_diffusion_ground_truth.csv",
        index_col=0,
    )

    playability_normal = df_normal.groupby("z")["marioStatus"].mean()
    playability_baseline = df_baseline.groupby("z")["marioStatus"].mean()
    playability_geometric = df_geometric.groupby("z")["marioStatus"].mean()

    print(playability_normal)
    print(playability_baseline)
    print(playability_geometric)
    print(f"Mean playability for normal: {np.mean(playability_normal)}")
    print(f"Mean playability for baseline: {np.mean(playability_baseline)}")
    print(f"Mean playability for geodesic: {np.mean(playability_geometric)}")


if __name__ == "__main__":
    n_clusters = 500
    vaeh = VAEGeometryHierarchical()
    vaeh.load_state_dict(t.load(f"./models/hierarchical_final_playable_final.pt"))
    vaeh.update_cluster_centers(beta=-3.5, n_clusters=n_clusters, only_playable=True)

    # figure_grid_levels(vaeh)
    # figure_metric_for_beta(vaeh, n_clusters=n_clusters)
    # figure_metric_for_different_betas(vaeh, n_clusters=n_clusters)
    # ground_truth_plot(vaeh)
    # save_interpolations(vaeh)
    # analyse_interpolation_results(vaeh)
    # compare_ground_truth_and_metric_volume(vaeh)
    # save_diffusion_experiment(vaeh)
    # analyse_diffusion_experiment(vaeh)
    # plot_saved_diffusions(vaeh)
    # run_diffusion_on_ground_truth(vaeh)
    analyse_diffusion_experiment_on_ground_truth(vaeh)

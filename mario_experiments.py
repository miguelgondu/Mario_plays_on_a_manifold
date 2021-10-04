from typing import Tuple

import torch as t
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from vae_mario_hierarchical import VAEMarioHierarchical
from vae_geometry_hierarchical import VAEGeometryHierarchical

from geoml.discretized_manifold import DiscretizedManifold

from interpolations.linear_interpolation import LinearInterpolation
from interpolations.geodesic_interpolation import GeodesicInterpolation

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

    # # Plot the first 5.
    # _, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    # # correctness = vae.get_correctness_img("syntactic", sample=True)
    # playability = get_ground_truth()
    # ax1.imshow(playability, extent=[-5, 5, -5, 5], cmap="Blues", vmin=0.0, vmax=1.0)
    # ax2.imshow(playability, extent=[-5, 5, -5, 5], cmap="Blues", vmin=0.0, vmax=1.0)
    # for line, geodesic in zip(fifty_lines, fifty_geodesics):
    #     line = line.detach().numpy()
    #     geodesic = geodesic.detach().numpy()

    #     # coh_line = expected_coherences_in_lines[i]
    #     # coh_geodesic = expected_coherences_in_geodesics[i]
    #     ax1.plot(line[:, 0], line[:, 1])
    #     ax2.plot(geodesic[:, 0], geodesic[:, 1])

    # ax1.set_title("Lines")
    # ax2.set_title("Geodesics")
    # plt.tight_layout()
    # plt.show()
    # plt.close()

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


if __name__ == "__main__":
    n_clusters = 500
    vaeh = VAEGeometryHierarchical()
    vaeh.load_state_dict(t.load(f"./models/hierarchical_final_playable_final.pt"))
    vaeh.update_cluster_centers(beta=-3.5, n_clusters=n_clusters)

    # figure_grid_levels(vaeh)
    # figure_metric_for_beta(vaeh, n_clusters=n_clusters)
    # figure_metric_for_different_betas(vaeh, n_clusters=n_clusters)
    # ground_truth_plot(vaeh)
    save_interpolations(vaeh)

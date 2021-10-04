import torch as t
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from vae_mario_hierarchical import VAEMarioHierarchical
from vae_geometry_hierarchical import VAEGeometryHierarchical

from metric_approximation_with_jacobians import approximate_metric, plot_approximation


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


def ground_truth_plot(vae):
    """
    Grabs the ground truth table and computes the average
    playability of each position. Compares with the latent space.
    """
    df = pd.read_csv(
        "./data/processed/ground_truth/hierarchical_final_playable_final_ground_truth.csv"
    )
    playability = df.groupby(["z1", "z2"])["marioStatus"].mean()
    # print(playability)
    # print(playability.index.values)

    # print(playability.loc[(-5.0, -5.0)])
    z1 = np.array(list(set([idx[0] for idx in playability.index.values])))
    z1 = np.sort(z1)
    z2 = np.array(list(set([idx[1] for idx in playability.index.values])))
    z2 = np.sort(z2)

    # playability_v = playability.values

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

    _, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.imshow(
        playability_img, extent=[z1.min(), z1.max(), z2.min(), z2.max()], cmap="Blues"
    )
    ax.axis("off")

    zs = vae.encodings.detach().numpy()
    ax.scatter(zs[:, 0], zs[:, 1], marker="x", c="#DC851F")
    plt.tight_layout()
    plt.savefig(
        f"./data/plots/final/mario_ground_truth.png", dpi=100, bbox_inches="tight"
    )
    plt.show()
    plt.close()


if __name__ == "__main__":
    n_clusters = 500
    vaeh = VAEGeometryHierarchical()
    vaeh.load_state_dict(t.load(f"./models/hierarchical_final_playable_final.pt"))
    vaeh.update_cluster_centers(beta=-3.0)

    # figure_grid_levels(vaeh)
    # figure_metric_for_beta(vaeh, n_clusters=n_clusters)
    # figure_metric_for_different_betas(vaeh, n_clusters=n_clusters)
    ground_truth_plot(vaeh)

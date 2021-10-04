import torch as t
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    n_clusters = 500
    vaeh = VAEGeometryHierarchical()
    vaeh.load_state_dict(t.load(f"./models/hierarchical_final_playable_final.pt"))
    vaeh.update_cluster_centers(beta=-3.0)

    # figure_grid_levels(vaeh)
    # figure_metric_for_beta(vaeh, n_clusters=n_clusters)
    figure_metric_for_different_betas(vaeh, n_clusters=n_clusters)

import torch as t
import matplotlib.pyplot as plt
from metric_approximation_with_jacobians import plot_approximation

from vae_text import VAEText
from vae_hierarchical_text import VAEHierarchicalText
from vae_geometry_text import VAEGeometryText
from vae_geometry_hierarchical_text import VAEGeometryHierarchicalText
from diffusions.geometric_difussion import GeometricDifussion


def plot_semantics_and_syntaxis():
    vaetext = VAEText()
    vaetext.load_state_dict(t.load(f"./models/test_text_final.pt"))

    semantic = vaetext.plot_correctness("semantic")
    syntactic = vaetext.plot_correctness("syntactic")

    _, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(syntactic, extent=[-5, 5, -5, 5], cmap="Blues")
    latent_codes, _ = vaetext.forward(vaetext.train_tensor)
    latent_codes = latent_codes.mean.detach().numpy()
    ax1.scatter(latent_codes[:, 0], latent_codes[:, 1], c="k", marker="x")

    ax2.imshow(semantic, extent=[-5, 5, -5, 5], cmap="Blues")
    ax2.scatter(latent_codes[:, 0], latent_codes[:, 1], c="k", marker="x")

    plt.show()

    vaetext.plot_correctness("semantic")


def geometry_plots():
    vaetext = VAEGeometryText()
    vaetext.load_state_dict(t.load(f"./models/test_text_final.pt"))
    vaetext.update_cluster_centers(beta=-1.5, n_clusters=300)
    gd = GeometricDifussion(50)
    syntactic = vaetext.plot_correctness("syntactic")

    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(syntactic, extent=[-5, 5, -5, 5], cmap="Blues")
    latent_codes, _ = vaetext.forward(vaetext.train_tensor)
    latent_codes = latent_codes.mean.detach().numpy()
    ax1.scatter(latent_codes[:, 0], latent_codes[:, 1], c="k", marker="x")

    vaetext.plot_w_geodesics(ax=ax2)

    ax3.set_xlim((-5, 5))
    ax3.set_ylim((-5, 5))
    vaetext.plot_latent_space(ax=ax3, plot_points=False)
    for _ in range(5):
        zs = gd.run(vaetext).detach().numpy()
        ax3.plot(zs[:, 0], zs[:, 1])
        ax3.scatter(zs[:1, 0], zs[:1, 1], c="c", marker="o", zorder=10)
        ax3.scatter(zs[:, 0], zs[:, 1], c="g", marker="x")

    plt.tight_layout()
    plt.show()


def geometry_plots_but_hierarchical():
    vaetext = VAEGeometryHierarchicalText()
    vaetext.load_state_dict(t.load(f"./models/hierarchical_text_final.pt"))
    vaetext.update_cluster_centers(beta=-0.5, n_clusters=500)
    gd = GeometricDifussion(50)
    syntactic = vaetext.plot_correctness("syntactic")

    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(syntactic, extent=[-5, 5, -5, 5], cmap="Blues")
    latent_codes, _ = vaetext.forward(vaetext.train_tensor)
    latent_codes = latent_codes.mean.detach().numpy()
    ax1.scatter(latent_codes[:, 0], latent_codes[:, 1], c="k", marker="x")
    vaetext.plot_w_geodesics(ax=ax2)

    # ax3.set_xlim((-5, 5))
    # ax3.set_ylim((-5, 5))
    # vaetext.plot_latent_space(ax=ax3, plot_points=False)
    # for _ in range(5):
    #     zs = gd.run(vaetext).detach().numpy()
    #     ax3.plot(zs[:, 0], zs[:, 1])
    #     ax3.scatter(zs[:1, 0], zs[:1, 1], c="c", marker="o", zorder=10)
    #     ax3.scatter(zs[:, 0], zs[:, 1], c="g", marker="x")

    plt.tight_layout()
    plt.show()


def several_betas():
    # Plotting one with no UQ.
    vaetext = VAEGeometryHierarchicalText()
    vaetext.load_state_dict(t.load(f"./models/hierarchical_text_final.pt"))
    _, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot_approximation(vaetext, function=vaetext.decode, ax=ax)
    # zs = vae.encodings.detach().numpy()
    # ax.scatter(zs[:, 0], zs[:, 1], marker="x", c="#DC851F")
    ax.set_title("No UQ", fontsize=20)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        f"./data/plots/final/text_model_no_UQ.png", dpi=100, bbox_inches="tight"
    )
    plt.close()

    for beta in [-1.0, -2.0, -3.0, -4.0]:
        vaetext = VAEGeometryHierarchicalText()
        vaetext.load_state_dict(t.load(f"./models/hierarchical_text_final.pt"))
        vaetext.update_cluster_centers(beta=beta, n_clusters=500)

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        vaetext.plot_metric_volume(ax=ax)
        ax.set_title(r"$\beta=$" + f"{beta}", fontsize=20)
        ax.axis("off")
        plt.tight_layout()
        fig.savefig(
            f"./data/plots/final/text_model_beta_{beta}.png",
            dpi=100,
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    # geometry_plots()
    # geometry_plots_but_hierarchical()
    several_betas()

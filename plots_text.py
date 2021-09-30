import torch as t
import matplotlib.pyplot as plt

from vae_text import VAEText
from vae_geometry_text import VAEGeometryText
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
        ax3.plot(zs[:, 0], zs[:, 1], "g")
        ax3.scatter(zs[:1, 0], zs[:1, 1], c="c", marker="o", zorder=10)
        ax3.scatter(zs[:, 0], zs[:, 1], c="g", marker="x")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    geometry_plots()

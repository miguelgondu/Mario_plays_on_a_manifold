import torch
import numpy as np
from torch.distributions import Categorical, Dirichlet
import matplotlib.pyplot as plt

from vae_geometry_base import VAEGeometryBase

Tensor = torch.Tensor


class VAEGeometryDirichlet(VAEGeometryBase):
    def __init__(
        self,
        w: int = 14,
        h: int = 14,
        z_dim: int = 2,
        n_sprites: int = 11,
        device: str = None,
    ):
        super().__init__(w, h, z_dim, n_sprites=n_sprites, device=device)

    def reweight(self, z: Tensor) -> Categorical:
        """
        Random dirichlet noise.
        """
        similarity = self.translated_sigmoid(self.min_distance(z)).view(-1, 1, 1, 1)
        dec_categorical = self.decode(z)
        dec_probs = dec_categorical.probs

        random_probs = Dirichlet(torch.ones_like(dec_probs)).sample()

        reweighted_probs = (1 - similarity) * dec_probs + similarity * (random_probs)
        p_x_given_z = Categorical(probs=reweighted_probs)

        return p_x_given_z


if __name__ == "__main__":
    model_name = "vae_deeper_lr_1e-4_no_overfit_final"
    vae = VAEGeometryDirichlet()
    vae.load_state_dict(torch.load(f"./models/{model_name}.pt", map_location="cpu"))
    beta = -3.5
    vae.update_cluster_centers(beta=beta, n_clusters=500)
    # _, (ax1, ax2) = plt.subplots(1, 2)
    # vae.plot_latent_space(ax=ax1)
    # vae.plot_w_geodesics(ax=ax2, plot_points=False)
    # plt.show()
    # """
    # Returns some plots for the circle dataset,
    # plotting geodescis and approximating metrics.
    # """
    # vae = Model()
    # vae.load_state_dict(t.load(f"models/{model_name}.pt", map_location="cpu"))
    # print("Updating cluster centers")
    # print(encodings)

    # Circle data
    # angles = torch.rand((100,)) * 2 * np.pi
    # encodings = 3.0 * torch.vstack((torch.cos(angles), torch.sin(angles))).T
    # vae.update_cluster_centers(beta=-2.5, encodings=encodings)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * 2, 10))
    x_lims = (-5, 5)
    y_lims = (-5, 5)

    print("Plotting geodesics and latent space")
    try:
        vae.plot_w_geodesics(ax=ax1)
    except Exception as e:
        print(f"couldn't get geodesics for reason {e}")

    n_x, n_y = 50, 50
    z1 = torch.linspace(*x_lims, n_x)
    z2 = torch.linspace(*y_lims, n_x)
    positions = {
        (x.item(), y.item()): (i, j)
        for j, x in enumerate(z1)
        for i, y in enumerate(reversed(z2))
    }

    zs = torch.Tensor([[x, y] for x in z1 for y in z2])
    metric_volume = np.zeros((n_y, n_x))
    for z in zs:
        (x, y) = z
        i, j = positions[(x.item(), y.item())]
        Mz = vae.metric(z)

        detMz = torch.det(Mz).item()
        if detMz < 0:
            metric_volume[i, j] = np.nan
        else:
            metric_volume[i, j] = np.log(detMz)

    cbar = ax2.imshow(metric_volume, extent=[*x_lims, *y_lims], cmap="Blues")
    plt.colorbar(cbar)

    ax1.set_title("Latent space and geodesics")
    ax2.set_title("Estimated metric volume")
    fig.suptitle(f"beta = {beta}")
    plt.tight_layout()
    plt.show()

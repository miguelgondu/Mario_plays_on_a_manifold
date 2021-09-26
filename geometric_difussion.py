import torch as t
from torch.distributions import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt

from vae_geometry_dirichlet import VAEGeometryDirichlet


def geometric_random_walk():
    model_name = "16324019946774652_mariovae_zdim_2_epoch_40"
    vae = VAEGeometryDirichlet()
    vae.load_state_dict(t.load(f"models/{model_name}.pt", map_location="cpu"))
    print("Updating cluster centers")
    # print(encodings)

    angles = t.rand((100,)) * 2 * np.pi
    encodings = 3.0 * t.vstack((t.cos(angles), t.sin(angles))).T
    vae.update_cluster_centers(
        model_name, False, beta=-2.5, encodings=encodings, n_clusters=100
    )

    n_steps = 100
    z_n = vae.encodings[0, :]
    z2_n = vae.encodings[0, :]
    zs = [z_n]
    zs_normal = [z2_n]
    for _ in range(n_steps):
        Mz = vae.metric(z_n)

        d = MultivariateNormal(z_n, covariance_matrix=Mz.inverse())
        z_n = d.rsample()
        zs.append(z_n)

        scale = 0.5
        d2 = MultivariateNormal(z2_n, covariance_matrix=scale * t.eye(2))
        z2_n = d2.rsample()
        zs_normal.append(z2_n)

    zs = t.vstack(zs)
    zs = zs.detach().numpy()

    zs_normal = t.vstack(zs_normal)
    zs_normal = zs_normal.detach().numpy()

    _, ax = plt.subplots(1, 1)

    vae.plot_latent_space(ax=ax)
    plt.scatter(zs[:, 0], zs[:, 1], c="g", marker="x")
    plt.scatter(zs_normal[:, 0], zs_normal[:, 1], c="r", marker="x")
    plt.show()

    return zs, zs_normal


geometric_random_walk()

"""
Computes geodesics on a circle for a given
model
"""

from typing import Union
import torch as t
import numpy as np
import matplotlib.pyplot as plt

from metric_approximation import MetricApproximation

from vae_geometry_base import VAEGeometryBase
from vae_geometry_dirichlet import VAEGeometryDirichlet
from vae_geometry_uniform import VAEGeometryUniform
from vae_geometry_hierarchical import VAEGeometryHierarchical


def circle_experiment(
    model_name: str,
    Model: VAEGeometryBase,
):
    """
    Returns some plots for the circle dataset,
    plotting geodescis and approximating metrics.
    """
    vae = Model()
    vae.load_state_dict(t.load(f"models/{model_name}.pt", map_location="cpu"))
    print("Updating cluster centers")
    # print(encodings)

    angles = t.rand((100,)) * 2 * np.pi
    encodings = 3.0 * t.vstack((t.cos(angles), t.sin(angles))).T
    vae.update_cluster_centers(model_name, False, beta=-2.5, encodings=encodings)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * 2, 10))
    x_lims = (-6, 6)
    y_lims = (-6, 6)

    print("Plotting geodesics and latent space")
    try:
        vae.plot_w_geodesics(ax=ax1, plot_points=False)
    except Exception as e:
        print(f"couldn't get geodesics for reason {e}")

    M = MetricApproximation(model=vae, z_dim=2, eps=0.05)

    n_x, n_y = 50, 50
    x_lims = (-6, 6)
    y_lims = (-6, 6)
    z1 = t.linspace(*x_lims, n_x)
    z2 = t.linspace(*y_lims, n_x)
    positions = {
        (x.item(), y.item()): (i, j)
        for j, x in enumerate(z1)
        for i, y in enumerate(reversed(z2))
    }

    zs = t.Tensor([[x, y] for x in z1 for y in z2])
    metric_volume = np.zeros((n_y, n_x))
    for z in zs:
        (x, y) = z
        i, j = positions[(x.item(), y.item())]
        Mz = M(z)

        detMz = t.det(Mz).item()
        if detMz < 0:
            metric_volume[i, j] = np.nan
        else:
            metric_volume[i, j] = np.log(detMz)

    cbar = ax2.imshow(metric_volume, extent=[*x_lims, *y_lims], cmap="Blues")
    plt.colorbar(cbar)

    ax1.set_title("Latent space and geodesics")
    ax2.set_title("Estimated metric volume")

    plt.tight_layout()
    plt.savefig(f"data/plots/circle_experiment_{model_name}.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    model_names = [
        "16324019946774652_mariovae_zdim_2_epoch_40",
        "final_overfitted_nnj_epoch_300",
        "mariovae_w_relu_epoch_180",
    ]

    models = [VAEGeometryDirichlet, VAEGeometryHierarchical, VAEGeometryUniform]

    for model, model_name in zip(models, model_names):
        circle_experiment(model_name, model)

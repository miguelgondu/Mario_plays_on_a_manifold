import torch as t
import numpy as np
import matplotlib.pyplot as plt

from vae_geometry_hierarchical import VAEGeometryHierarchical
from vae_geometry_dirichlet import VAEGeometryDirichlet

from metric_approximation import MetricApproximation


def plots(vae, model_name):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(7 * 2, 7))
    vae.plot_w_geodesics(ax=ax1)
    # plt.show()

    M = MetricApproximation(model=vae, z_dim=2, eps=0.05)
    # z = t.Tensor([0.0, 0.0])

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
    for l, z in enumerate(zs):
        (x, y) = z
        i, j = positions[(x.item(), y.item())]
        Mz = M(z)
        # print(Mz)
        detMz = t.det(Mz).item()
        if detMz < 0:
            metric_volume[i, j] = np.nan
        else:
            metric_volume[i, j] = np.log(detMz)

    cbar = ax2.imshow(metric_volume, extent=[*x_lims, *y_lims], cmap="Blues")
    plt.colorbar(cbar)
    plt.tight_layout()
    plt.savefig(f"./data/plots/metric_volume_{model_name}.png")
    plt.show()


def approximate_metric_hierarchical():
    model_name = "final_overfitted_nnj_epoch_120"

    vae = VAEGeometryHierarchical()
    vae.load_state_dict(t.load(f"models/{model_name}.pt", map_location="cpu"))
    vae.update_cluster_centers(model_name, False, beta=-2.5, n_clusters=2000)
    print("Updated cluster centers.")

    plots(vae, model_name)


if __name__ == "__main__":
    model_name = "mariovae_w_relu_epoch_180"
    # vae = VAEMario()
    vae = VAEGeometryDirichlet()
    vae.load_state_dict(t.load(f"./models/{model_name}.pt"))
    vae.update_cluster_centers(model_name, False, beta=-2.5, n_clusters=500)

    plots(vae, model_name)

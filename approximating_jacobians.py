from vae_geometry_base import VAEGeometryBase
import torch
import numpy as np
import matplotlib.pyplot as plt

from vae_geometry_dirichlet import VAEGeometryDirichlet


def fd_jacobian(function, x, h=1e-4):
    """
    Compute finite difference Jacobian of given function
    at a single location x. This function is mainly considered
    useful for debugging."""

    no_batch = x.dim() == 1
    if no_batch:
        x = x.unsqueeze(0)
    elif x.dim() > 2:
        raise Exception("The input should be a D-vector or a BxD matrix")
    B, D = x.shape

    # Compute finite differences
    E = h * torch.eye(D)
    Jnum = torch.cat(
        [
            (
                (
                    function(x[b] + E).probs.view(-1, 14 * 14 * 11)
                    - function(x[b].unsqueeze(0)).probs.view(-1, 14 * 14 * 11)
                ).t()
                / h
            ).unsqueeze(0)
            for b in range(B)
        ]
    )

    if no_batch:
        Jnum = Jnum.squeeze(0)

    return Jnum


def approximate_metric(function, z, h=0.01):
    J = fd_jacobian(function, z, h=h)
    if len(J.shape) > 2:
        return torch.bmm(J.transpose(1, 2), J)
    else:
        return J.T @ J


def plot_approximation(model: VAEGeometryBase):
    n_x, n_y = 50, 50
    x_lims = (-6, 6)
    y_lims = (-6, 6)
    z1 = torch.linspace(*x_lims, n_x)
    z2 = torch.linspace(*y_lims, n_x)
    positions = {
        (x.item(), y.item()): (i, j)
        for j, x in enumerate(z1)
        for i, y in enumerate(reversed(z2))
    }

    zs = torch.Tensor([[x, y] for x in z1 for y in z2])
    metric_volume = np.zeros((n_y, n_x))
    metrics = approximate_metric(model.reweight, zs)
    for z, Mz in zip(zs, metrics):
        (x, y) = z
        i, j = positions[(x.item(), y.item())]
        # print(Mz)
        detMz = torch.det(Mz).item()
        if detMz < 0:
            metric_volume[i, j] = np.nan
        else:
            metric_volume[i, j] = np.log(detMz)

    _, (ax1, ax2) = plt.subplots(1, 2)
    model.plot_latent_space(ax=ax1, plot_points=False)

    ax2.imshow(metric_volume, extent=[*x_lims, *y_lims], cmap="Blues")

    ax1.set_title("Latent space w. Entropy")
    ax2.set_title("Numerical approximation of the metric")
    plt.show()


if __name__ == "__main__":
    # Load the model
    vae = VAEGeometryDirichlet()
    vae.load_state_dict(
        torch.load("./models/mariovae_w_relu_epoch_160.pt", map_location="cpu")
    )
    vae.update_cluster_centers(
        **{
            "beta": -1.5,
            "n_clusters": 500,
            "only_playable": False,
        }
    )

    plot_approximation(vae)

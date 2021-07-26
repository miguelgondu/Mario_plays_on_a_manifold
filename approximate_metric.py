import torch
from torch.distributions import Distribution, Categorical
import matplotlib.pyplot as plt
import numpy as np

from vae_geometry import VAEGeometry
from mario_utils.plotting import plot_level_from_array, plot_level_from_decoded_tensor

Tensor = torch.Tensor


def compute_distance(model: VAEGeometry, z1: Tensor, z2: Tensor):
    """
    Computes the KL after decoding z1 and z2
    with model.
    """
    # dec_z = model.reweight(z1)[0]
    # dec_z_dz = model.reweight(z2)[0]

    # p = Categorical(logits=dec_z.transpose(1, 3))
    # q = Categorical(logits=dec_z_dz.transpose(1, 3))
    # return torch.distributions.kl_divergence(p, q).mean().item()
    zs = torch.vstack((z1, z2))
    return model.curve_length(zs).item()


def local_KL(model: VAEGeometry, zs, eps=1e-2):
    """
    Approximates KL locally for all points z in zs,
    and returns mean KL around each z. These are computed
    by adding eps * [[1,0], [0,1], [-1,0], [0,-1]] to
    z, computing the respective KLs after decoding and taking
    the mean.
    """
    dzs = (
        eps
        * torch.Tensor(
            [
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
            ]
        ).type(torch.float)
    )

    KLs = []
    for z in zs:
        KLs_for_z = [compute_distance(model, z, z + dz) for dz in dzs]
        # print(KLs_for_z)
        KLs.append(np.mean(KLs_for_z))

    return KLs


if __name__ == "__main__":
    model_name = "mariovae_z_dim_2_overfitting_epoch_100"
    vae = VAEGeometry()
    vae.load_state_dict(torch.load(f"models/{model_name}.pt"))
    vae.update_cluster_centers(model_name, False, beta=-1.5)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(7 * 2, 7))

    # KLs = local_KL(vae, torch.Tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), eps=0.05)
    # print(KLs)

    n_x, n_y = 50, 50
    x_lims = (-6, 6)
    y_lims = (-6, 6)
    z1 = torch.linspace(*x_lims, n_x)
    z2 = torch.linspace(*y_lims, n_x)

    KL_image = np.zeros((n_y, n_x))
    zs = torch.Tensor([[x, y] for x in z1 for y in z2])
    positions = {
        (x.item(), y.item()): (i, j)
        for j, x in enumerate(z1)
        for i, y in enumerate(reversed(z2))
    }

    KLs = local_KL(vae, zs, eps=0.05)
    for l, (x, y) in enumerate(zs):
        i, j = positions[(x.item(), y.item())]
        KL_image[i, j] = KLs[l]

    # encodings = vae.encodings.detach().numpy()
    # ax.scatter(
    #     encodings[:, 0],
    #     encodings[:, 1],
    #     marker="o",
    #     c="w",
    #     edgecolors="k",
    # )
    ax2.scatter(vae.cluster_centers[:, 0], vae.cluster_centers[:, 1], marker="x", c="k")
    plot = ax2.imshow(KL_image, extent=[*x_lims, *y_lims], cmap="viridis")
    plt.colorbar(plot, ax=ax2)

    vae.plot_w_geodesics(ax=ax1)
    plt.show()

import torch
import matplotlib.pyplot as plt
import numpy as np

from vae_geometry import VAEGeometry
from mario_utils.levels import onehot_to_levels
from mario_utils.plotting import get_img_from_level

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


def plot_grid_reweight(vae, ax, x_lims, y_lims, n_rows=10, n_cols=10, title=""):
    z1 = np.linspace(*x_lims, n_cols)
    z2 = np.linspace(*y_lims, n_rows)

    zs = torch.Tensor([[a, b] for b in reversed(z2) for a in z1])
    images = vae.reweight(zs)[0]
    images = onehot_to_levels(images.detach().numpy())
    images = np.array([get_img_from_level(im) for im in images])
    zs = zs.detach().numpy()
    # print(zs)
    final_img = np.vstack(
        [
            np.hstack([im for im in row])
            for row in images.reshape((n_cols, n_rows, 16 * 14, 16 * 14, 3))
        ]
    )
    ax.imshow(final_img, extent=[*x_lims, *y_lims])
    # ax.imshow(final_img)
    # ax.set_title(f"Decoded samples ({title})")


if __name__ == "__main__":
    model_name = "mariovae_z_dim_2_overfitting_epoch_480"
    vae = VAEGeometry()
    vae.load_state_dict(torch.load(f"models/{model_name}.pt"))
    print("Updating cluster centers")
    vae.update_cluster_centers(model_name, False, beta=-1.5)
    # raise

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7 * 3, 7))

    print("Plotting grid of levels")
    x_lims = (-6, 6)
    y_lims = (-6, 6)
    plot_grid_reweight(vae, ax1, x_lims, y_lims, n_rows=12, n_cols=12)

    print("Plotting geodesics and latent space")
    vae.plot_w_geodesics(ax=ax2)

    print("Plotting Local KL approximation")
    n_x, n_y = 50, 50
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

    ax3.scatter(vae.cluster_centers[:, 0], vae.cluster_centers[:, 1], marker="x", c="k")
    plot = ax3.imshow(KL_image, extent=[*x_lims, *y_lims], cmap="viridis")
    # plt.colorbar(plot, ax=ax3)

    ax1.set_title("Decoded levels")
    ax2.set_title("Latent space and geodesics")
    ax3.set_title("Estimated metric volume")

    plt.tight_layout()
    plt.savefig("data/plots/geodesics.png")

    plt.show()

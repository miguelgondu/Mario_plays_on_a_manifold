import torch
import numpy as np
import matplotlib.pyplot as plt

from vae_mario import VAEMario

from mario_utils.levels import onehot_to_levels
from mario_utils.plotting import get_img_from_level


def plot_grid(vae, ax, x_lims, y_lims, n_rows=10, n_cols=10, title=""):
    z1 = np.linspace(*x_lims, n_cols)
    z2 = np.linspace(*y_lims, n_rows)

    zs = torch.Tensor([[a, b] for a in reversed(z1) for b in z2])
    images = vae.decoder(zs)
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
    ax.set_title(f"Decoded samples ({title})")


if __name__ == "__main__":
    vae = VAEMario(14, 14, z_dim=2)
    model_names = [
        # "mariovae_z_dim_2_only_playable_epoch_480",
        "mariovae_z_dim_2_overfitting_epoch_480",
    ]
    for model_name in model_names:
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        vae.load_state_dict(torch.load(f"./models_experiment/{model_name}.pt"))
        vae.eval()

        title = "only_playable" if "playable" in model_name else "normal"
        plot_grid(vae, ax, [-15, 15], [-15, 15], n_rows=30, n_cols=30, title=title)
        # ax.axis("off")
        fig.savefig(
            f"./data/plots/grid_of_levels_{model_name}_bigger_15.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

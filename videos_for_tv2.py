import gc
import multiprocessing as mp

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

matplotlib.use("TkAgg")

from vae_mario import VAEMario
from train_vae import load_data, optim, TensorDataset, DataLoader, fit, test

from mario_utils.plotting import get_img_from_level


def plot_images(z, images, ax):
    """
    A function that plots all images in {images}
    at coordinates {z}.
    """
    for zi, img in zip(z, images):
        im = OffsetImage(img, zoom=0.5)
        ab = AnnotationBbox(im, zi, xycoords="data", frameon=True)
        ax.add_artist(ab)
        ax.update_datalim([zi])
        ax.autoscale()


# Loading the data.


def plot_image(vae, epoch, training_tensors, og_imgs):
    n_rows = n_cols = 10
    _, ax1 = plt.subplots(1, 1, figsize=(7 * n_rows, 7 * n_cols))
    print(f"Plotting reconstructions ({epoch})")
    q_z_given_x, p_x_given_z = vae.forward(training_tensors)
    levels = p_x_given_z.probs.argmax(dim=-1).detach().numpy()
    imgs = [get_img_from_level(l) for l in levels]
    zs = q_z_given_x.mean.detach().numpy()
    plot_images(zs, imgs, ax1)
    ax1.set_xlim((-5, 5))
    ax1.set_ylim((-5, 5))
    ax1.axis("off")
    plt.tight_layout()
    plt.savefig(f"./data/plots/imgs_for_videos/reconstructions_{epoch:05d}.png")
    plt.close()
    print(f"Saved reconstructions for epoch {epoch}.")
    del imgs

    print(f"Plotting grid ({epoch})")
    _, ax2 = plt.subplots(1, 1, figsize=(7 * n_rows, 7 * n_cols))
    vae.plot_grid(ax=ax2)
    plt.tight_layout()
    plt.savefig(f"./data/plots/imgs_for_videos/grid_{epoch:05d}.png")
    plt.close()
    print(f"Saved grid for epoch {epoch}.")

    _, ax3 = plt.subplots(1, 1, figsize=(7 * n_rows, 7 * n_cols))
    print(f"Plotting embeddings ({epoch})")
    plot_images(zs, og_imgs, ax3)
    ax3.set_xlim((-5, 5))
    ax3.set_ylim((-5, 5))
    plt.tight_layout()
    plt.savefig(f"./data/plots/imgs_for_videos/embeddings_{epoch:05d}.png")
    plt.close()
    print(f"Saved embeddings for epoch {epoch}.")
    # del og_imgs

    gc.collect()


if __name__ == "__main__":
    playable = False
    seed = 0
    z_dim = 2
    training_tensors, _ = load_data(shuffle_seed=seed, only_playable=playable)
    vae = VAEMario(z_dim=z_dim)
    og_levels = training_tensors.argmax(dim=1).detach().numpy()
    og_imgs = [get_img_from_level(og_l) for og_l in og_levels]
    for epoch in list(range(1, 191)).__reversed__():
        print(f"Epoch {epoch}")
        vae.load_state_dict(torch.load(f"./models/mariovae_videos_epoch_{epoch}.pt"))
        print(f"Loaded model ({epoch})")
        plot_image(vae, epoch, training_tensors, og_imgs)
        gc.collect()

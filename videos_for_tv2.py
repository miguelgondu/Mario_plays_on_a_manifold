import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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


playable = False
batch_size = 64
seed = 0
max_epochs = 300
overfit = True
save_every = 20
z_dim = 2
lr = 1e-4

# Setting up the hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the data.
training_tensors, test_tensors = load_data(shuffle_seed=seed, only_playable=playable)

# Creating datasets.
dataset = TensorDataset(training_tensors)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(test_tensors)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Loading the model
print("Model:")
vae = VAEMario(z_dim=z_dim)
optimizer = optim.Adam(vae.parameters(), lr=lr)

# Training and testing.
levels_for_reconstruction = test_tensors[:2, :, :, :].detach().numpy()
best_loss = np.Inf
n_without_improvement = 0
img_id = 0
for epoch in range(max_epochs):
    print(f"Epoch {epoch + 1} of {max_epochs}.")
    train_loss = fit(vae, optimizer, data_loader, device)
    test_loss = test(vae, test_loader, test_dataset, device, epoch)
    if test_loss < best_loss:
        best_loss = test_loss
        n_without_improvement = 0
    else:
        if not overfit:
            n_without_improvement += 1

    if epoch % 3 == 0:
        img_id += 1
        n_rows = n_cols = 10
        fig1, ax1 = plt.subplots(1, 1, figsize=(7 * n_rows, 7 * n_cols))
        ax1.set_xlim((-6, 6))
        ax1.set_ylim((-6, 6))
        q_z_given_x, p_x_given_z = vae.forward(training_tensors)
        levels = p_x_given_z.probs.argmax(dim=-1).detach().numpy()
        imgs = [get_img_from_level(l) for l in levels]
        zs = q_z_given_x.mean.detach().numpy()
        plot_images(zs, imgs, ax1)
        ax1.axis("off")
        plt.tight_layout()
        plt.savefig(f"./data/plots/imgs_for_videos/reconstructions_{img_id:05d}.png")
        plt.close()

        fig2, ax2 = plt.subplots(1, 1, figsize=(7 * n_rows, 7 * n_cols))
        vae.plot_grid(ax=ax2)
        plt.tight_layout()
        plt.savefig(f"./data/plots/imgs_for_videos/grid_{img_id:05d}.png")
        plt.close()

        fig3, ax3 = plt.subplots(1, 1, figsize=(7 * n_rows, 7 * n_cols))
        ax3.set_xlim((-6, 6))
        ax3.set_ylim((-6, 6))
        og_levels = training_tensors.argmax(dim=1).detach().numpy()
        og_imgs = [get_img_from_level(og_l) for og_l in og_levels]
        plot_images(zs, og_imgs, ax3)
        plt.tight_layout()
        plt.savefig(f"./data/plots/imgs_for_videos/embeddings_{img_id:05d}.png")
        plt.close()

    # Early stopping:
    if n_without_improvement == 10:
        print("Stopping early")
        break

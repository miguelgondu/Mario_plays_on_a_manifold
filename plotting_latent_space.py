import torch
import numpy as np
import matplotlib.pyplot as plt

from mario_utils.plotting import get_img_from_level
from mario_utils.levels import onehot_to_levels
from vae_mario import VAEMario
from train_vae import load_data

# plt.rcParams.update({'font.size': 22})


def plot_latent_space(model, training_data, val_data):
    # Loading up the training points
    zs_training, logvars_training = model.encode(training_data)
    zs_val, logvars_val = model.encode(val_data)

    print("zs for data")
    print(zs_training)
    print(zs_training.shape)

    print("zs for test")
    print(zs_val)
    print(zs_val.shape)

    zs_training = zs_training.detach().numpy()
    zs_val = zs_val.detach().numpy()
    logvars_training: np.ndarray = logvars_training.detach().numpy().sum(axis=1)
    logvars_val: np.ndarray = logvars_val.detach().numpy().sum(axis=1)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.scatter(zs_training[:, 0], zs_training[:, 1], label="Training set")
    ax1.scatter(zs_val[:, 0], zs_val[:, 1], label="Validation set")
    ax1.legend()

    vars_training = np.exp(logvars_training)
    vars_val = np.exp(logvars_val)
    lower_bound = np.min(np.concatenate((vars_training, vars_val)))
    upper_bound = np.max(np.concatenate((vars_training, vars_val)))
    ax2.scatter(
        zs_training[:, 0],
        zs_training[:, 1],
        c=vars_training,
        vmin=lower_bound,
        vmax=upper_bound,
    )
    plot = ax2.scatter(
        zs_val[:, 0], zs_val[:, 1], c=vars_val, vmin=lower_bound, vmax=upper_bound
    )
    plt.colorbar(plot, ax=ax2)
    plt.show()

    # n_rows = 10
    # n_cols = 10
    # z1 = np.linspace(5, -5, n_cols)
    # z2 = np.linspace(5, -5, n_rows)

    # zs = torch.Tensor([
    #     [a, b] for a in z1 for b in z2
    # ])
    # images = model.decoder(zs)
    # images = onehot_to_levels(images.detach().numpy())
    # images = np.array([get_img_from_level(im) for im in images])
    # # print(images.shape)
    # zs = zs.detach().numpy()
    # # print(zs)
    # final_img = np.vstack([np.hstack([im for im in row]) for row in images.reshape((n_cols, n_rows, 256, 256, 3))])
    # _, ax = plt.subplots(1, 1, figsize=(3*n_cols, 3*n_rows))
    # ax.imshow(final_img)
    # ax.set_title(f"Latent Space for Smooth VAE (Epoch {checkpoint})")
    # plt.xticks(np.linspace(0, final_img.shape[0], 10), map(lambda x: f"{x:1.1f}", np.linspace(-5, 5, 10)))
    # plt.yticks(np.linspace(0, final_img.shape[1], 10), map(lambda x: f"{x:1.1f}", np.linspace(-5, 5, 10)))
    # # plt.show()
    # plt.tight_layout()
    # plt.savefig(f"./data/latent_spaces/vae_mario_z_dim_2_smooth_epoch_{checkpoint:04d}.png", dpi=150)
    # plt.close()


def zero_log(level: np.ndarray):
    zero_log_level = np.copy(level)
    zero_log_level[np.where(level == 0)] = 1
    return np.log(zero_log_level)


def mean_entropy(level: np.ndarray):
    # Assuming it is a [K, w, h] array
    level[:, 0, 0] = [1 / 12] * 12
    H = level * zero_log(level)
    H = -np.sum(H, axis=0)  # over classes.
    return np.mean(H)


def main():
    # Loading up the model
    model_name = "mariovae_z_dim_2_only_playable_epoch_480"
    vae = VAEMario(14, 14, z_dim=2)
    vae.load_state_dict(torch.load(f"./models/{model_name}.pt"))
    vae.eval()

    training_tensors, test_tensors = load_data(only_playable=True)
    test_levels = test_tensors.detach().numpy()
    # for level in test_levels:
    #     print(mean_entropy(level))
    plot_latent_space(vae, training_tensors, test_tensors)


if __name__ == "__main__":
    main()

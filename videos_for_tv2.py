import torch
import numpy as np
import matplotlib.pyplot as plt

from vae_mario import VAEMario
from train_vae import load_data, optim, TensorDataset, DataLoader, fit, test

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

    n_rows = n_cols = 10
    _, ax = plt.subplots(1, 1, figsize=(7 * n_rows, 7 * n_cols))
    vae.plot_grid(ax=ax)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"./data/plots/imgs_for_videos/{epoch:05d}.png", dpi=150)
    plt.close()

    # Early stopping:
    if n_without_improvement == 10:
        print("Stopping early")
        break

"""
This script trains a VAESimple on the Mario
levels.
"""
import json
from time import time

import click
import torch
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
from tqdm import tqdm

from vae_mario import VAEMario
from torch.utils.tensorboard import SummaryWriter

# Data types.
Tensor = torch.Tensor


def load_data(
    training_percentage=0.8, test_percentage=None, shuffle_seed=0, only_playable=False
):
    """Returns two tensors with training and testing data"""
    # Loading the data.
    # This data is structured [b, c, i, j], where c corresponds to the class.
    if only_playable:
        data = np.load("./data/processed/all_playable_levels_onehot.npz")["levels"]
    else:
        data = np.load("./data/processed/all_levels_onehot.npz")["levels"]

    # if only_playable:
    #     np.random.seed(0)
    #     np.random.shuffle(data)

    #     with open("./data/processed/playable_levels_idxs.json") as fp:
    #         playable_level_idxs = json.load(fp)

    #     data = data[playable_level_idxs]
    # else:
    #     np.random.seed(shuffle_seed)

    np.random.seed(shuffle_seed)
    np.random.shuffle(data)

    # Separating into training and test.
    n_data, _, _, _ = data.shape
    training_index = int(n_data * training_percentage)
    training_data = data[:training_index, :, :, :]
    testing_data = data[training_index:, :, :, :]
    training_tensors = torch.from_numpy(training_data)
    test_tensors = torch.from_numpy(testing_data)
    training_tensors = training_tensors.type(torch.FloatTensor)
    test_tensors = test_tensors.type(torch.FloatTensor)

    return training_tensors, test_tensors


# Next step: defining the loss function.
# Cross Entropy + KLD regularization
def fit(
    model: VAEMario,
    optimizer: Optimizer,
    data_loader: DataLoader,
    device: str,
):
    model.train()
    running_loss = 0.0
    for _, levels in tqdm(enumerate(data_loader)):
        levels = levels[0]
        levels = levels.to(device)
        optimizer.zero_grad()
        q_z_given_x, p_x_given_z = model.forward(levels)
        loss = model.elbo_loss_function(levels, q_z_given_x, p_x_given_z)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    return running_loss


def test(
    model: VAEMario,
    test_loader: DataLoader,
    test_dataset: Dataset,
    device: str,
    epoch: int = 0,
):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for _, levels in tqdm(enumerate(test_loader)):
            levels = levels[0]
            levels.to(device)
            q_z_given_x, p_x_given_z = model.forward(levels)
            loss = model.elbo_loss_function(levels, q_z_given_x, p_x_given_z)
            running_loss += loss.item()

    print(f"Epoch {epoch}. Loss in test: {running_loss / len(test_dataset)}")
    return running_loss


@click.command()
@click.option("--z-dim", type=int, default=2)
@click.option("--comment", type=str, default=None)
@click.option("--max-epochs", type=int, default=200)
@click.option("--batch-size", type=int, default=64)
@click.option("--lr", type=float, default=1e-3)
@click.option("--seed", type=int, default=0)
@click.option("--scale", type=float, default=1.0)
@click.option("--save-every", type=int, default=20)
@click.option("--overfit/--no-overfit", default=False)
@click.option("--playable/--no-playable", default=False)
def run(
    z_dim,
    comment,
    max_epochs,
    batch_size,
    lr,
    seed,
    scale,
    save_every,
    overfit,
    playable,
):
    # Setting up the seeds
    torch.manual_seed(seed)

    # Defining the name of the experiment
    timestamp = str(time()).replace(".", "")
    if comment is None:
        comment = f"{timestamp}_mariovae_zdim_{z_dim}"

    writer = SummaryWriter(log_dir=f"./runs/{timestamp}_{comment}")

    # Setting up the hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the data.
    training_tensors, test_tensors = load_data(
        shuffle_seed=seed, only_playable=playable
    )

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
    print(f"Training experiment {comment}")
    best_loss = np.Inf
    n_without_improvement = 0
    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1} of {max_epochs}.")
        train_loss = fit(vae, optimizer, data_loader, device)
        test_loss = test(vae, test_loader, test_dataset, device, epoch)
        if test_loss < best_loss:
            best_loss = test_loss
            n_without_improvement = 0

            # Saving the best model so far.
            torch.save(vae.state_dict(), f"./models/{comment}_final.pt")
        else:
            if not overfit:
                n_without_improvement += 1

        # Reporting
        vae.report(
            writer,
            epoch,
            train_loss / len(dataset),
            test_loss / len(test_dataset),
        )

        if epoch % save_every == 0 and epoch != 0:
            # Saving the model
            print(f"Saving the model at checkpoint {epoch}.")
            torch.save(vae.state_dict(), f"./models/{comment}_epoch_{epoch}.pt")

        # Early stopping:
        if n_without_improvement == 10:
            print("Stopping early")
            break


if __name__ == "__main__":
    run()
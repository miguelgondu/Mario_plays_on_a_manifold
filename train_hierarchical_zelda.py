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

from shapeguard import ShapeGuard

from vae_zelda_hierarchical import VAEZeldaHierarchical, load_data
from torch.utils.tensorboard import SummaryWriter

# Data types.
Tensor = torch.Tensor


# Next step: defining the loss function.
# Cross Entropy + KLD regularization
def fit(
    model: VAEZeldaHierarchical,
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
    model: VAEZeldaHierarchical,
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
@click.option("--save-every", type=int, default=20)
@click.option("--overfit/--no-overfit", default=False)
def run(
    z_dim,
    comment,
    max_epochs,
    batch_size,
    lr,
    seed,
    save_every,
    overfit,
):
    # Setting up the seeds
    torch.manual_seed(seed)

    # Defining the name of the experiment
    timestamp = str(time()).replace(".", "")
    if comment is None:
        comment = f"{timestamp}_zeldavae_zdim_{z_dim}"

    writer = SummaryWriter(log_dir=f"./runs/{timestamp}_{comment}")

    # Setting up the hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the data.
    training_tensors, test_tensors = load_data(shuffle_seed=seed)

    # Creating datasets.
    dataset = TensorDataset(training_tensors)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_tensors)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Loading the model
    print("Model:")
    vae = VAEZeldaHierarchical()
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # Training and testing.
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

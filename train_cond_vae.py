"""
This script trains a CondVAEMario on the Mario
levels.
"""
import json
from time import time
from typing import List

import click
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mario_utils.levels import onehot_to_levels
from mario_utils.plotting import save_level_from_array
from mario_utils.plotting import plot_level_from_array
from mario_utils.plotting import get_img_from_level
from cond_vae_mario import CondVAEMario
from train_vae import load_data
from torch.utils.tensorboard import SummaryWriter

# Data types.
Tensor = torch.Tensor


def load_data_w_classes() -> List[Tensor]:
    training_tensors, test_tensors = load_data()
    all_levels = torch.cat((training_tensors, test_tensors))
    df = pd.read_csv("./data/processed/training_levels_results.csv", index_col=0)
    playability = df.groupby("idx")["marioStatus"].mean()
    playable_idxs = playability.index[playability > 0]

    all_classes = np.zeros((all_levels.shape[0]))
    all_classes[playable_idxs.values] = 1.0
    all_classes = torch.from_numpy(all_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        all_levels, all_classes, test_size=0.33, random_state=0
    )

    # print(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


# Next step: defining the loss function.
# Cross Entropy + KLD regularization
def loss_function(x_prime, x, mu, log_var, scale=1.0):
    # Assmuing that both are 1-hot encoding representations.
    x_classes = onehot_to_levels(x)
    loss = torch.nn.NLLLoss(reduction="sum")
    CEL = loss(x_prime, x_classes)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return CEL + scale * KLD, CEL, KLD


def plot_samples(vae, zs, comment=None):
    _, axes = plt.subplots(2, 2, figsize=(2 * 7, 2 * 7))
    axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    samples = vae.decode(zs, torch.ones((zs.shape[0],)))
    print(f"Samples shape: {samples.shape}")
    samples = onehot_to_levels(samples.detach().numpy())
    print(f"Samples shape: {samples.shape}")
    for level, ax in zip(samples, axes):
        # print(level)
        plot_level_from_array(ax, level)

    plt.tight_layout()
    plt.savefig(f"./data/samples/samples_{comment}.png", bbox_inches="tight")
    plt.close()


def plot_reconstructions(vae, levels, classes, comment=None):
    _, axes = plt.subplots(2, 2, figsize=(2 * 7, 2 * 7))
    reconst, _, _, _ = vae(torch.Tensor(levels), classes)
    levels = onehot_to_levels(levels)
    levels_prime = onehot_to_levels(reconst.detach().numpy())
    for i, (level, level_prime) in enumerate(zip(levels[:2], levels_prime[:2])):
        ax1, ax2 = axes[i, :]

        if i == 0:
            ax1.set_title("Original")
            ax2.set_title("Reconstruction")

        # Plotting level
        ax1.imshow(255 * np.ones_like(level))
        ax1.imshow(get_img_from_level(level))
        ax1.axis("off")

        # Plotting level reconstruction
        ax2.imshow(255 * np.ones_like(level_prime))
        ax2.imshow(get_img_from_level(level_prime))
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig(
        f"./data/reconstructions/reconstruction_{comment}.png", bbox_inches="tight"
    )
    plt.close()


def fit(model, optimizer, data_loader, dataset, device, writer, epoch=0, scale=1.0):
    model.train()
    running_loss = 0.0
    CELs = []
    KLDs = []
    for i, (levels, classes) in tqdm(enumerate(data_loader)):
        levels = levels.to(device)
        classes = classes.to(device)
        optimizer.zero_grad()
        x_primes, xs, mu, log_var = model(levels, classes)
        loss, CEL, KLD = loss_function(x_primes, xs, mu, log_var, scale=scale)
        running_loss += loss.item()
        CELs.append(CEL.item() / len(levels))
        KLDs.append(KLD.item() / len(levels))
        loss.backward()
        optimizer.step()
        # if i % 2 == 0:
        #     report(writer, )
    report(writer, "train", epoch, loss.item() / len(dataset))

    return running_loss


def test(model, test_loader, test_dataset, device, writer, epoch=0, scale=1.0):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (levels, classes) in tqdm(enumerate(test_loader)):
            levels.to(device)
            classes.to(device)
            x_primes, xs, mu, log_var = model(levels, classes)
            loss, _, _ = loss_function(x_primes, xs, mu, log_var, scale=scale)
            running_loss += loss.item()

    report(writer, "test", epoch, loss.item() / len(test_dataset))
    print(f"Epoch {epoch}. Loss in test: {running_loss / len(test_dataset)}")
    return running_loss


def report(writer: SummaryWriter, comment: str, checkpoint: int, loss: float):
    # writer.add_scalar(f"KLD - {comment}", KLD, checkpoint)
    # writer.add_scalar(f"CEL - {comment}", CEL, checkpoint)
    writer.add_scalar(f"loss - {comment}", loss, checkpoint)

    # samples = decoder(zs)
    # samples = onehot_to_levels(
    #     samples.detach().numpy()
    # )
    # samples = np.array(
    #     [get_img_from_level(level) for level in samples]
    # )

    # writer.add_images(f"samples_{epoch}", samples, checkpoint, dataformats="NHWC")


@click.command()
@click.option("--z-dim", type=int, default=32)
@click.option("--h-dim", type=int, default=None, multiple=True)
@click.option("--comment", type=str, default=None)
@click.option("--max-epochs", type=int, default=200)
@click.option("--batch-size", type=int, default=64)
@click.option("--lr", type=float, default=1e-3)
@click.option("--seed", type=int, default=0)
@click.option("--scale", type=float, default=1.0)
@click.option("--save", type=int, default=20)
@click.option("--overfit/--no-overfit", default=False)
@click.option("--playable/--no-playable", default=False)
def run(
    z_dim,
    h_dim,
    comment,
    max_epochs,
    batch_size,
    lr,
    seed,
    scale,
    save,
    overfit,
    playable,
):
    # Setting up the seeds
    torch.manual_seed(seed)

    # Defining the name of the experiment
    timestamp = str(time()).replace(".", "")
    if comment is None:
        comment = f"{timestamp}_cond_mariovae_zdim_{z_dim}"

    writer = SummaryWriter(log_dir=f"./runs/{comment}")
    # Setting up the hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the data.
    X_train, X_test, y_train, y_test = load_data_w_classes()
    print("-" * 25 + " Training data " + "-" * 25)
    print(f"X train: {X_train.shape} ({X_train.dtype})")
    print(f"y train: {y_train.shape} ({y_train.dtype})")
    print("-" * 25 + " Testing data " + "-" * 25)
    print(f"X test: {X_test.shape} ({X_test.dtype})")
    print(f"y test: {y_test.shape} ({y_test.dtype})")

    # -----------------------------------------------------------------
    ## Overfitting test: check if we overfit to one batch.
    # overfitting_data = data[:64, :, :, :]
    # test_overfitting_data = data[64:64*2, :, :, :]
    # training_tensors = torch.from_numpy(overfitting_data)
    # test_tensors = torch.from_numpy(test_overfitting_data)
    # training_tensors = training_tensors.type(torch.FloatTensor)
    # test_tensors = test_tensors.type(torch.FloatTensor)
    # -----------------------------------------------------------------

    # Creating datasets.
    dataset = TensorDataset(X_train, y_train)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if isinstance(h_dim, int):
        h_dim = [h_dim]
    if isinstance(h_dim, tuple):
        h_dim = list(h_dim)

    print(h_dim)
    # Loading the model
    w = h = 14
    vae = CondVAEMario(w, h, z_dim, h_dims=h_dim)  # z_dim, h_dim come via click.
    optimizer = optim.Adam(vae.parameters())
    zs = torch.randn(4, z_dim)

    # Training and testing.
    levels_for_reconstruction = X_test[:2, :, :, :].detach().numpy()
    classes_for_reconstruction = y_test[:2].detach().numpy()
    print(f"Training experiment {comment}")
    train_losses = []
    test_losses = []
    best_loss = np.Inf
    n_without_improvement = 0
    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1} of {max_epochs}.")
        train_loss = fit(
            vae,
            optimizer,
            data_loader,
            dataset,
            device,
            writer,
            epoch=epoch,
            scale=scale,
        )
        train_losses.append(train_loss / len(dataset))
        test_loss = test(
            vae, test_loader, test_dataset, device, writer, epoch=epoch, scale=scale
        )
        test_losses.append(test_loss / len(test_dataset))
        if test_loss < best_loss:
            best_loss = test_loss
            n_without_improvement = 0

            # Saving the best model so far.
            torch.save(vae.state_dict(), f"./models/{comment}_final.pt")
        else:
            if not overfit:
                n_without_improvement += 1

        if epoch % save == 0 and epoch != 0:
            # Saving the model
            print("Plotting samples and reconstructions")
            plot_samples(vae, zs, comment=f"{comment}_epoch_{epoch}")
            plot_reconstructions(
                vae, levels_for_reconstruction, comment=f"{comment}_epoch_{epoch}"
            )

            print(f"Saving the model at checkpoint {epoch}.")
            torch.save(vae.state_dict(), f"./models/{comment}_epoch_{epoch}.pt")

        # Early stopping:
        if n_without_improvement == 10:
            print("Stopping early")
            break

    results = {"train_losses": train_losses, "test_losses": test_losses}
    # upload_blob_from_dict("training_experiments", results, f"{comment}.json")

    with open(f"./data/training_results/training_and_test_{comment}.json", "w") as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    run()
    # load_data_w_classes()

import json
from pathlib import Path
from typing import List
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
from torch.distributions import Distribution, Normal, Categorical, kl_divergence
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from grammar_zelda import grammar_check

from zelda_utils.plotting import encoding, get_img_from_level


def load_data() -> t.Tensor:
    training_percentage = 0.9

    levels = np.load("./data/processed/zelda/onehot.npz")["levels"]
    levels = np.array([l for l in levels if grammar_check(l.argmax(axis=-1))])
    np.random.shuffle(levels)

    n_data, _, _, _ = levels.shape
    training_index = int(n_data * training_percentage)
    training_data = levels[:training_index, :, :, :]
    testing_data = levels[training_index:, :, :, :]
    training_tensors = t.from_numpy(training_data).type(t.float)
    test_tensors = t.from_numpy(testing_data).type(t.float)
    return training_tensors, test_tensors


class VAEZeldaHierarchical(nn.Module):
    def __init__(self, z_dim: int = 2):
        super(VAEZeldaHierarchical, self).__init__()

        self.train_data, _ = load_data()
        _, h, w, n_sprites = self.train_data.shape
        self.w = w
        self.h = h
        self.n_sprites = n_sprites
        self.input_dim = w * h * n_sprites  # for flattening
        self.z_dim = z_dim
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        ).to(self.device)
        self.enc_mu = nn.Sequential(nn.Linear(128, z_dim)).to(self.device)
        self.enc_var = nn.Sequential(nn.Linear(128, z_dim)).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, self.input_dim),
            nn.Tanh(),
        ).to(self.device)
        self.dec_mu = nn.Linear(self.input_dim, self.input_dim).to(self.device)
        self.dec_var = nn.Linear(self.input_dim, self.input_dim).to(self.device)

        self.p_z = Normal(
            t.zeros(self.z_dim, device=self.device),
            t.ones(self.z_dim, device=self.device),
        )

        # print(self)

    def encode(self, x: t.Tensor) -> Normal:
        # Returns q(z | x) = Normal(mu, sigma)
        x = x.view(-1, self.input_dim).to(self.device)
        result = self.encoder(x)
        mu = self.enc_mu(result)
        log_var = self.enc_var(result)

        return Normal(mu, t.exp(0.5 * log_var))

    def _intermediate_distribution(self, z: t.Tensor) -> Normal:
        result = self.decoder(z.to(self.device))
        dec_mu = self.dec_mu(result)
        dec_log_var = self.dec_var(result)

        return Normal(dec_mu, t.exp(0.5 * dec_log_var))

    def decode(self, z: t.Tensor) -> Categorical:
        # Returns p(x | z) = Cat(logits=samples from _intermediate_distribution)
        dec_dist = self._intermediate_distribution(z.to(self.device))
        samples = dec_dist.rsample()
        p_x_given_z = Categorical(
            logits=samples.reshape(-1, self.h, self.w, self.n_sprites)
        )

        return p_x_given_z

    def forward(self, x: t.Tensor) -> List[Distribution]:
        q_z_given_x = self.encode(x.to(self.device))

        z = q_z_given_x.rsample()

        p_x_given_z = self.decode(z.to(self.device))

        return [q_z_given_x, p_x_given_z]

    def elbo_loss_function(
        self, x: t.Tensor, q_z_given_x: Distribution, p_x_given_z: Distribution
    ) -> t.Tensor:
        x_ = x.to(self.device).argmax(dim=-1)  # assuming x is bchw.
        rec_loss = -p_x_given_z.log_prob(x_).sum(dim=(1, 2))  # b
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1)  # b

        return (rec_loss + kld).mean()

    def random_sample(self, fig_name: str = None) -> t.Tensor:
        z = t.randn((64, 2))
        levels = self.decode(z).probs.argmax(dim=-1)

        if fig_name is not None:
            fig, axes = plt.subplots(8, 8, figsize=(8 * 7, 8 * 7))
            for level, ax in zip(levels.detach().numpy(), axes.flatten()):
                img = get_img_from_level(level)
                ax.imshow(255 * np.ones_like(img))
                ax.imshow(img)
                ax.axis("off")
            fig.set_facecolor("white")
            plt.tight_layout()
            plt.savefig(
                f"./data/plots/zelda/random_samples/{fig_name}.png",
                dpi=100,
                bbox_inches="tight",
            )
            plt.close()
            # plt.show()

        return levels

    def plot_grid(
        self,
        x_lims=(-5, 5),
        y_lims=(-5, 5),
        n_rows=10,
        n_cols=10,
        ax=None,
    ):
        z1 = np.linspace(*x_lims, n_cols)
        z2 = np.linspace(*y_lims, n_rows)

        zs = np.array([[a, b] for a, b in product(z1, z2)])

        images_dist = self.decode(t.from_numpy(zs).type(t.float))
        images = images_dist.probs.argmax(dim=-1)

        images = np.array(
            [get_img_from_level(im) for im in images.cpu().detach().numpy()]
        )
        img_dict = {(z[0], z[1]): img for z, img in zip(zs, images)}

        positions = {
            (x, y): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
        }

        lvl_height = images[0].shape[0]
        lvl_width = images[0].shape[1]

        final_img = 255 * np.ones((n_cols * lvl_height, n_rows * lvl_width, 4))
        for z, (i, j) in positions.items():
            final_img[
                i * (lvl_height) : (i + 1) * (lvl_height),
                j * (lvl_width) : (j + 1) * (lvl_width),
                :,
            ] = img_dict[z]

        final_img = final_img.astype(int)

        if ax is not None:
            ax.imshow(final_img, extent=[*x_lims, *y_lims])

        return final_img


def fit(
    model: VAEZeldaHierarchical, optimizer: t.optim.Optimizer, data_loader: DataLoader
):
    model.train()
    running_loss = 0.0
    for _, levels in tqdm(enumerate(data_loader)):
        levels = levels[0]
        levels = levels.to(model.device)
        optimizer.zero_grad()
        q_z_given_x, p_x_given_z = model.forward(levels)
        loss = model.elbo_loss_function(levels, q_z_given_x, p_x_given_z)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    return running_loss / len(data_loader)


def test(
    model: VAEZeldaHierarchical,
    test_loader: DataLoader,
    epoch: int = 0,
):
    model.eval()
    running_loss = 0.0
    with t.no_grad():
        for _, levels in tqdm(enumerate(test_loader)):
            levels = levels[0]
            levels.to(model.device)
            q_z_given_x, p_x_given_z = model.forward(levels)
            loss = model.elbo_loss_function(levels, q_z_given_x, p_x_given_z)
            running_loss += loss.item()

    print(f"Epoch {epoch}. Loss in test: {running_loss / len(test_loader)}")
    return running_loss / len(test_loader)


def run(id_: int = 0):
    # Setting up the seeds
    # torch.manual_seed(seed)
    batch_size = 64
    lr = 1e-4
    comment = "zelda_hierarchical"
    max_epochs = 250
    overfit = False
    save_every = 20

    # Loading the data.
    training_tensors, test_tensors = load_data()

    # Creating datasets.
    dataset = TensorDataset(training_tensors)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_tensors)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Loading the model
    print("Model:")
    vae = VAEZeldaHierarchical()

    print(vae)
    optimizer = t.optim.Adam(vae.parameters(), lr=lr)

    # Training and testing.
    print(f"Training experiment {comment}")
    best_loss = np.Inf
    n_without_improvement = 0
    losses = {"train": [], "test": []}
    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1} of {max_epochs}.")
        train_loss = fit(vae, optimizer, data_loader)
        test_loss = test(vae, test_loader, epoch=epoch)
        losses["train"].append(train_loss)
        losses["test"].append(test_loss)
        if test_loss < best_loss:
            best_loss = test_loss
            n_without_improvement = 0

            # Saving the best model so far.
            t.save(vae.state_dict(), f"./models/zelda/{comment}_final_{id_}_n.pt")
        else:
            n_without_improvement += 1

        if epoch % save_every == 0 and epoch != 0:
            # Saving the model
            print(f"Saving the model at checkpoint {epoch}.")
            t.save(vae.state_dict(), f"./models/zelda/{comment}_epoch_{epoch}.pt")

        # Early stopping:
        if n_without_improvement == 25 and not overfit:
            print("Stopping early")
            break

    with open(f"./data/training_results/zelda/{comment}_final_{id_}.json", "w") as fp:
        json.dump(losses, fp)


def plot_grid_of_levels(vae_path: Path):
    x_lims = (-4, 4)
    y_lims = (-4, 4)

    vae = VAEZeldaHierarchical()
    vae.load_state_dict(t.load(vae_path, map_location=vae.device))

    grid = vae.plot_grid(x_lims=x_lims, y_lims=y_lims, n_rows=10, n_cols=10)

    fig, ax = plt.subplots(1, 1, figsize=(15 * 7, 11 * 7))
    ax.imshow(grid, extent=[*x_lims, *y_lims])
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        f"./data/plots/zelda/grids/{vae_path.stem}.png", dpi=100, bbox_inches="tight"
    )
    plt.close(fig)


def plot_grammar_in_latent_space(vae_path: Path, force: bool = False):
    model_name = vae_path.stem
    g_path = Path(f"./data/processed/zelda/grammar_checks/{model_name}.npz")
    vae = VAEZeldaHierarchical()
    vae.load_state_dict(t.load(vae_path))
    x_lims = (-4, 4)
    y_lims = (-4, 4)
    n_rows = n_cols = 50
    z1 = np.linspace(*x_lims, n_cols)
    z2 = np.linspace(*y_lims, n_rows)
    positions = {
        (x, y): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
    }
    zs_in_positions = t.Tensor([z for z in positions.keys()]).type(t.float)

    if g_path.exists() and not force:
        ps = np.load(g_path)["playabilities"]
    else:
        levels = vae.decode(zs_in_positions).probs.argmax(dim=-1)
        ps = [int(grammar_check(level)) for level in levels]

    grammar_img = np.zeros((n_cols, n_rows))
    for (_, pos), p in zip(positions.items(), ps):
        grammar_img[pos] = int(p)

    encodings = vae.encode(vae.train_data).mean.detach().numpy()
    np.savez(
        f"./data/processed/zelda/grammar_checks/{model_name}.npz",
        zs=zs_in_positions.detach().numpy(),
        playabilities=np.array(ps).astype(float),
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(grammar_img, extent=[*x_lims, *y_lims], cmap="Blues")
    ax.scatter(encodings[:, 0], encodings[:, 1])
    ax.axis("off")
    fig.savefig(
        f"./data/plots/zelda/ground_truths/{model_name}.png",
        dpi=100,
        bbox_inches="tight",
    )
    # plt.show()
    plt.close(fig)


def plot_random_samples(vae_path: Path):
    vae = VAEZeldaHierarchical()
    vae.load_state_dict(t.load(vae_path))
    vae.random_sample(fig_name=vae_path.stem)


if __name__ == "__main__":
    # train
    # for id_ in range(5):
    #     run(id_)

    # inspect
    # for path_ in Path("./models/zelda").glob("zelda_hierarchical_final_*.pt"):
    #     print(f"Processing {path_}")
    #     plot_grammar_in_latent_space(path_)
    #     plot_grid_of_levels(path_)
    #     plot_random_samples(path_)

    # vae = VAEZeldaHierarchical()
    # vae.load_state_dict(t.load("./models/zelda/zelda_hierarchical_final_0.pt"))
    # vae.random_sample()

    # x_lims = (-3, 3)
    # y_lims = (-3, 3)
    # grid = vae.plot_grid(x_lims=x_lims, y_lims=y_lims, n_rows=10, n_cols=10)
    # _, ax = plt.subplots(1, 1, figsize=(15 * 7, 11 * 7))
    # ax.imshow(grid, extent=[*x_lims, *y_lims])
    # ax.axis("off")

    # plt.tight_layout()
    # plt.savefig("./data/plots/zelda/grid.png", dpi=100, bbox_inches="tight")
    # plt.close()
    load_data()

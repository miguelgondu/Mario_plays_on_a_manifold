"""
Taking the MLP model from the MarioGAN paper and
training it on my database, in a comparable way
to the hVAE.
"""
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import Dataset
from utils.mario.plotting import get_img_from_level

from vae_models.vae_mario_hierarchical import load_data


class MLP_G(nn.Module):
    def __init__(
        self,
        img_size: int = 14,
        z_dim: int = 2,
        n_sprites: int = 11,
    ):
        super(MLP_G, self).__init__()
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.input_dim = img_size * img_size * n_sprites
        self.generator = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, n_sprites * img_size * img_size),
            # nn.Softmax(dim=-1)
        ).to(self.device)
        self.n_sprites = n_sprites
        self.img_size = img_size
        self.z_dim = z_dim

    def forward(self, z: t.Tensor) -> t.Tensor:
        b, _ = z.shape
        logits = self.generator(z.to(self.device)).view(
            b, self.img_size, self.img_size, self.n_sprites
        )
        softmax = nn.Softmax(dim=-1)
        levels = softmax(logits)

        return levels


class MLP_D(nn.Module):
    def __init__(
        self,
        img_size: int = 14,
        n_sprites: int = 11,
    ):
        super(MLP_D, self).__init__()

        self.input_dim = n_sprites * img_size * img_size
        self.img_size = img_size
        self.n_sprites = n_sprites
        self.classifier = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, level: t.Tensor) -> t.Tensor:
        b, _, _, _ = level.shape
        predictions = self.classifier(
            level.view(b, self.n_sprites * self.img_size * self.img_size)
        )

        return predictions


def prediction_error(predictions: t.Tensor, real: t.Tensor) -> t.Tensor:
    loss = -real * t.log(predictions) - (1 - real) * t.log(1 - predictions)

    return loss.mean()


def train_gan():
    n_epochs = 200
    # n_iters_d = 10
    n_iters_g = 15
    batch_size = 64 * 2
    train_tensors, test_tensors = load_data(only_playable=True)
    train_tensors = train_tensors.permute(0, 2, 3, 1)
    test_tensors = test_tensors.permute(0, 2, 3, 1)

    dataset = TensorDataset(train_tensors[:batch_size])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_tensors)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    discriminator = MLP_D()
    generator = MLP_G()

    optimizier_d = t.optim.Adam(discriminator.parameters(), lr=1e-4)
    optimizier_g = t.optim.Adam(generator.parameters(), lr=1e-4)

    epochs_discriminator = epochs_generator = 0

    for epoch in range(n_epochs):
        for i, (train_batch,) in enumerate(data_loader):
            if epochs_generator < 5:
                n_iters_d = 100
            else:
                n_iters_d = 5

            for _ in range(n_iters_d):
                # Updating the discriminator's weight
                discriminator.zero_grad()

                # On real data
                predictions_on_real = discriminator.forward(train_batch)
                error_on_real = prediction_error(
                    predictions_on_real, t.ones_like(predictions_on_real)
                )
                error_on_real.backward()

                # On fake data
                z = t.randn((batch_size, generator.z_dim))
                fake_levels = generator.forward(z)
                predictions_on_fake = discriminator.forward(fake_levels)
                error_on_fake = prediction_error(
                    predictions_on_fake, t.zeros_like(predictions_on_fake)
                )
                error_on_fake.backward()

                # error_discr = error_on_real + error_on_fake
                optimizier_d.step()

            epochs_discriminator += 1

            # Updating the generator's weight
            for _ in range(n_iters_g):
                z_2 = t.randn((batch_size, generator.z_dim))
                fake_levels_for_gen = generator.forward(z_2)
                predictions_on_fake_for_gen = discriminator(fake_levels_for_gen)

                generator.zero_grad()
                error_generator = prediction_error(
                    predictions_on_fake_for_gen,
                    t.ones_like(predictions_on_fake_for_gen),
                )
                error_generator.backward()

                optimizier_g.step()

            epochs_generator += 1

            if i % 2 == 0:
                print(
                    f"epoch: {epoch}, i: {i}, errorDreal: {error_on_real}, errorDfake: {error_on_fake}, errorG: {error_generator}."
                )

        if epoch % 5 == 0:
            t.save(
                discriminator.state_dict(),
                f"./models/MarioGAN/discriminator_{epoch}.pt",
            )
            t.save(generator.state_dict(), f"./models/MarioGAN/generator_{epoch}.pt")


def inspect_generators():
    """
    Loads and plots random samples from GANs
    """
    generator = MLP_G()
    for epoch in range(0, 200, 5):
        generator.load_state_dict(t.load(f"./models/MarioGAN/generator_{epoch}.pt"))
        z = 3.0 * t.randn((64, 2))
        samples = generator(z).argmax(dim=-1)
        _, axes = plt.subplots(8, 8, figsize=(7 * 8, 7 * 8))
        for ax, lvl in zip(axes.flatten(), samples):
            ax.imshow(get_img_from_level(lvl.detach().numpy()))
            ax.axis("off")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"./data/plots/MarioGAN/generator_{epoch}.png", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    train_gan()
    inspect_generators()

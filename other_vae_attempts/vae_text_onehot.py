"""
A version of the equation VAE that decodes to a OneHotCategorical.
The aim is to use the Gumbel distribution after, and modulate
the temperature as a way to ensure uncertainty quantification.
"""
import random
from typing import List
from itertools import product
from time import time

import numpy as np
import torch as t
from torch.distributions import (
    Distribution,
    Normal,
    RelaxedOneHotCategorical,
    OneHotCategorical,
    kl_divergence,
)
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

from train_vae_text import fit, test


def generate(max_terms, symbols="+-*"):
    n_terms = random.randint(2, max_terms)
    terms = [str(random.randint(0, 9)) for _ in range(n_terms)]

    eq = ""
    for term in terms:
        eq += term + random.choice(symbols)
    eq = eq[:-1]
    eq += "=" + str(eval(eq))
    return eq


def parse_semantically(eq_string: str):
    if eq_string.count("=") != 1:
        return False
    try:
        return eval(eq_string.replace("=", "=="))
    except:
        return False


def parse_syntactically(eq_string: str):
    if eq_string.count("=") != 1:
        return False
    try:
        _ = eval(eq_string.replace("=", "=="))
        return True
    except Exception as e:
        return False


def load_data(n_sequences: int, max_length: int = 10, seed=0):
    random.seed(seed)
    seqs = list(set([generate(4) for _ in range(n_sequences)]))
    seqs = [s for s in seqs if len(s) < max_length]
    while len(seqs) < n_sequences:
        seqs += [generate(4) for _ in range(n_sequences - len(seqs))]
        seqs = list(set(seqs))
        seqs = [s for s in seqs if len(s) < max_length]
        # print(seqs)

    seqs = [s.ljust(10) for s in seqs]
    idx = int(len(seqs) * 0.8)
    train_seqs, test_seqs = seqs[:idx], seqs[idx:]

    return train_seqs, test_seqs


class VAEGumbelText(nn.Module):
    def __init__(
        self,
        length: int = 10,
        z_dim: int = 2,
        device: str = None,
        temperature: float = 2.2,
    ):
        super(VAEGumbelText, self).__init__()
        self.encoding = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "=": 10,
            "+": 11,
            "-": 12,
            "*": 13,
            " ": 14,
        }
        self.inv_encoding = {v: k for k, v in self.encoding.items()}
        self.length = length
        self.n_symbols = len(self.encoding)
        self.input_dim = length * self.n_symbols
        self.z_dim = z_dim
        self.temperature = temperature
        self.device = device or t.device("cuda" if t.cuda.is_available() else "cpu")

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.Tanh(),
        ).to(self.device)
        self.enc_mu = nn.Sequential(nn.Linear(64, z_dim)).to(self.device)
        self.enc_var = nn.Sequential(nn.Linear(64, z_dim)).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 64),
            nn.Tanh(),
            nn.Linear(64, self.input_dim),
            nn.Tanh(),
        ).to(self.device)

        self.p_z = Normal(
            t.zeros(self.z_dim, device=self.device),
            t.ones(self.z_dim, device=self.device),
        )

        self.train_seqs, self.test_seqs = load_data(n_sequences=5000)

        # One-hot encode these sequences
        self.train_tensor = t.zeros((len(self.train_seqs), self.length, self.n_symbols))
        for b, seq in enumerate(self.train_seqs):
            for s, char in enumerate(seq):
                encoding = self.encoding[char]
                self.train_tensor[b, s, encoding] = 1.0

        self.test_tensor = t.zeros((len(self.test_seqs), self.length, self.n_symbols))
        for b, seq in enumerate(self.test_seqs):
            for s, char in enumerate(seq):
                encoding = self.encoding[char]
                self.test_tensor[b, s, encoding] = 1.0

    def int_sequence_to_text(self, x: t.Tensor) -> str:
        return "".join([self.inv_encoding[xi.item()] for xi in x])

    def encode(self, x: t.Tensor) -> Normal:
        # Returns q(z | x) = Normal(mu, sigma)
        x = x.view(-1, self.input_dim).to(self.device)
        result = self.encoder(x)
        mu = self.enc_mu(result)
        log_var = self.enc_var(result)

        return Normal(mu, t.exp(0.5 * log_var))

    def decode(self, z: t.Tensor) -> RelaxedOneHotCategorical:
        # Decodes z, returning p(x|z)
        b, _ = z.shape
        logits = self.decoder(z)

        p_x_given_z = OneHotCategorical(
            logits=logits.reshape(b, self.length, self.n_symbols),
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
        # x_ = x.to(self.device).argmax(dim=-1)  # assuming x is blc.
        rec_loss = -p_x_given_z.log_prob(x.to(self.device)).sum(dim=(1))  # b
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1)  # b

        return (rec_loss + kld).mean()

    def decode_to_text(self, z: t.Tensor) -> List[str]:
        samples_dist = self.decode(z)
        seqs_encoded = samples_dist.probs.argmax(dim=-1)
        return [
            "".join([self.inv_encoding[s.item()] for s in seq_encoded])
            for seq_encoded in seqs_encoded
        ]

    def report(
        self,
        writer: SummaryWriter,
        step_id: int,
        train_loss: float,
        test_loss: float,
    ):
        writer.add_scalar("train loss", train_loss, step_id)
        writer.add_scalar("test loss", test_loss, step_id)

        zs = self.p_z.sample((5,)).to(self.device)
        samples_dist = self.decode(zs)
        seqs_encoded = samples_dist.probs.argmax(dim=-1)
        print("Example sequences")
        for seq_encoded in seqs_encoded:
            seq = "".join([self.inv_encoding[s.item()] for s in seq_encoded])
            print(seq)

        syntactic_correctness = self.get_correctness_img("syntactic")
        writer.add_image(
            "syntactic correctness", syntactic_correctness, step_id, dataformats="HW"
        )

    def get_correctness_img(
        self, _type: str, x_lims=(-5, 5), y_lims=(-5, 5), n_x=50, n_y=50, sample=False
    ) -> np.ndarray:
        z1 = np.linspace(*x_lims, n_x)
        z2 = np.linspace(*y_lims, n_y)

        if _type == "semantic":
            corr = parse_semantically
        elif _type == "syntactic":
            corr = parse_syntactically
        else:
            raise ValueError("Expected semantic or syntactic")

        correctness_image = np.zeros((n_y, n_x))
        zs = t.Tensor([[x, y] for x in z1 for y in z2])
        positions = {
            (x.item(), y.item()): (i, j)
            for j, x in enumerate(z1)
            for i, y in enumerate(reversed(z2))
        }

        if sample:
            correctness = []
            for z in zs:
                dist = self.decode(z)
                samples = dist.sample((100,))
                sequences_in_samples = [
                    self.int_sequence_to_text(s[0]) for s in samples
                ]
                coherences_at_z = [corr(seq) for seq in sequences_in_samples]
                correctness.append(np.mean(coherences_at_z))
        else:
            sequences = self.decode_to_text(zs)
            correctness = [int(corr(seq)) for seq in sequences]

        for l, (x, y) in enumerate((product(z1, z2))):
            i, j = positions[(x.item(), y.item())]
            correctness_image[i, j] = correctness[l]

        return correctness_image


def run():
    # Hyperparameters
    z_dim = 2
    batch_size = 64
    lr = 0.002
    max_epochs = 1000
    overfit = True
    save_every = 100

    # Logging
    comment = f"vae_text_onehot_zdim_{z_dim}"
    timestamp = str(time()).replace(".", "")
    writer = SummaryWriter(log_dir=f"./runs/{timestamp}_{comment}")

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # Loading the model
    print("Model:")
    vae = VAEGumbelText(z_dim=z_dim, device=device, temperature=5.0)
    optimizer = t.optim.Adam(vae.parameters(), lr=lr)

    # Creting the datasets
    dataset = TensorDataset(vae.train_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(vae.test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

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
            t.save(vae.state_dict(), f"./models/text/{comment}_final.pt")
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
            t.save(vae.state_dict(), f"./models/text/{comment}_epoch_{epoch}.pt")

        # Early stopping:
        if n_without_improvement == 25:
            print("Stopping early")
            break


if __name__ == "__main__":
    # load_data(100)
    run()

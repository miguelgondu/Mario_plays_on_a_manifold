"""
We consider something like a one-layer hierarchical VAE
to see if this solves the problem of having high cost outside
the support.        
"""
import random
from datetime import datetime
from typing import List
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import _corrcoef_dispatcher
import torch as t
from torch.distributions import Distribution, Normal, Categorical, kl_divergence
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


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
    """
    Returns {n_sequences} unique equations of length at most {max_length}.
    The ones that are not of length {max_length} are padded with spaces.
    """
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


class VAEText(nn.Module):
    def __init__(
        self,
        length: int = 10,
        z_dim: int = 2,
        device: str = None,
        seed: int = 0,
    ):
        super(VAEText, self).__init__()
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
        ).to(self.device)

        self.p_z = Normal(
            t.zeros(self.z_dim, device=self.device),
            t.ones(self.z_dim, device=self.device),
        )

        self.train_seqs, self.test_seqs = load_data(n_sequences=5000, seed=seed)

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

        # print(self)

    def encode(self, x: t.Tensor) -> Normal:
        # Returns q(z | x) = Normal(mu, sigma)
        x = x.view(-1, self.input_dim).to(self.device)
        result = self.encoder(x)
        mu = self.enc_mu(result)
        log_var = self.enc_var(result)

        return Normal(mu, t.exp(0.5 * log_var))

    def decode(self, z: t.Tensor) -> Categorical:
        # Decodes z, returning p(x|z)
        logits = self.decoder(z.to(self.device))
        p_x_given_z = Categorical(
            logits=logits.reshape(-1, self.length, self.n_symbols)
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
        x_ = x.to(self.device).argmax(dim=-1)  # assuming x is blc.
        rec_loss = -p_x_given_z.log_prob(x_).sum(dim=(1))  # b
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1)  # b

        return (rec_loss + kld).mean()

    def decode_to_text(self, z: t.Tensor) -> List[str]:
        samples_dist = self.decode(z)
        seqs_encoded = samples_dist.probs.argmax(dim=-1)
        return [
            "".join([self.inv_encoding[s.item()] for s in seq_encoded])
            for seq_encoded in seqs_encoded
        ]

    def int_sequence_to_text(self, x: t.Tensor) -> str:
        return "".join([self.inv_encoding[xi.item()] for xi in x])

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


if __name__ == "__main__":
    load_data(100)

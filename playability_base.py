import json
from typing import List
from itertools import product
from time import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch as t
from torch.distributions import Bernoulli
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from shapeguard import ShapeGuard
from tqdm import tqdm


def get_level_datasets(random_state=0) -> List[TensorDataset]:
    """
    Returns train and test datasets.
    """

    # Load levels and playabilities
    df = pd.read_csv("./data/array_simulation_results/sampling.csv", index_col=0)
    mean_p_per_l = df.groupby(["level"])["marioStatus"].mean()
    # print(f"Unique levels: {len(mean_p_per_l.index)}")

    levels = []
    playabilities = []
    for l, p in mean_p_per_l.iteritems():
        levels.append(json.loads(l))
        playabilities.append(p)

    levels = np.array(levels)
    playabilities = np.array(playabilities)
    # Losing some information.
    playabilities[playabilities > 0.0] = 1.0

    b, h, w = levels.shape
    levels_onehot = np.zeros((b, 11, h, w))
    for batch, level in enumerate(levels):
        for i, j in product(range(h), range(w)):
            c = level[i, j]
            levels_onehot[batch, c, i, j] = 1.0

    l_train, l_test, p_train, p_test = train_test_split(
        levels_onehot, playabilities, random_state=random_state, stratify=playabilities
    )

    train_dataset = TensorDataset(
        t.from_numpy(l_train).type(t.float),
        t.from_numpy(p_train).type(t.float).view(-1, 1),
    )
    test_dataset = TensorDataset(
        t.from_numpy(l_test).type(t.float),
        t.from_numpy(p_test).type(t.float).view(-1, 1),
    )

    return train_dataset, test_dataset


def get_val_data() -> List[t.Tensor]:
    df = pd.read_csv("./data/processed/training_levels_results.csv")
    df_levels = df.groupby("level")["marioStatus"].mean()

    # print(f"Unique levels: {len(df_levels.index)}")

    levels = []
    playabilities = []
    for l, p in df_levels.iteritems():
        levels.append(json.loads(l))
        playabilities.append(p)

    levels = np.array(levels)[:, :, 1:]
    playabilities = np.array(playabilities)
    # Losing some information.
    playabilities[playabilities > 0.0] = 1.0

    b, h, w = levels.shape
    levels_onehot = np.zeros((b, 11, h, w))
    for batch, level in enumerate(levels):
        for i, j in product(range(h), range(w)):
            c = int(level[i, j])
            levels_onehot[batch, c, i, j] = 1.0

    return [
        t.from_numpy(levels_onehot).type(t.float),
        t.from_numpy(playabilities).type(t.float),
    ]


class PlayabilityBase(nn.Module):
    def __init__(self, batch_size: int = 64, random_state: int = 0):
        """
        An MLP used to predict playability of SMB levels.
        Adapted from the code I wrote for Rasmus' paper.
        """
        super(PlayabilityBase, self).__init__()
        self.w = 14
        self.h = 14
        self.n_classes = 11
        self.input_dim = 14 * 14 * 11  # for flattening
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")

        train_dataset, test_dataset = get_level_datasets(random_state=self.random_state)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        self.val_l, self.val_p = get_val_data()

        # This assumes that the data comes as 11x14x14.
        self.logits = None

        self.to(self.device)

    def forward(self, x: t.Tensor) -> Bernoulli:
        # Returns p(y | x) = Bernoulli(x; self.logits(x))
        raise NotImplementedError

    def binary_cross_entropy_loss(
        self, y: t.Tensor, p_y_given_x: Bernoulli
    ) -> t.Tensor:
        pred_loss = -p_y_given_x.log_prob(y).squeeze(1).sg("B")
        return pred_loss.mean()

    def report(self, writer: SummaryWriter, train_loss, test_loss, batch_id):
        writer.add_scalar("Mean Prediction Loss - Train", train_loss, batch_id)
        writer.add_scalar("Mean Prediction Loss - Test", test_loss, batch_id)

        # TODO: add reporting stats for validation
        ## - confusion matrix
        ## - histogram of predictions
        ## - plot comparing against ground truth in latent space.

        bern = self.forward(self.val_l)

        # Confusion matrix
        preds = bern.probs.detach().numpy()
        preds[preds > 0.5] = 1.0
        preds[preds <= 0.5] = 0.0
        C = confusion_matrix(self.val_p.detach().numpy(), preds)
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        sns.heatmap(C, ax=ax, annot=True)
        writer.add_figure("Confusion matrix (0.5 decision boundary)", fig, batch_id)

        # Histogram of probabilities
        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7))
        sns.histplot(data=bern.probs.detach().numpy(), ax=ax2)
        ax2.set_xlim((-0.05, 1.05))
        writer.add_figure("Histogram of probabilities", fig2, batch_id)


def fit(model: PlayabilityBase, optimizer: t.optim.Optimizer):
    model.train()
    running_loss = 0.0
    for (levels, p) in tqdm(model.train_loader):
        levels = levels.to(model.device)
        p = p.to(model.device)
        optimizer.zero_grad()
        p_y_given_x = model.forward(levels)
        loss = model.binary_cross_entropy_loss(p, p_y_given_x)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    return running_loss / len(model.train_loader)


def test(model: PlayabilityBase, epoch: int):
    model.eval()
    running_loss = 0.0
    with t.no_grad():
        for (levels, p) in tqdm(model.test_loader):
            levels = levels.to(model.device)
            p = p.to(model.device)
            p_y_given_x = model.forward(levels)
            loss = model.binary_cross_entropy_loss(p, p_y_given_x)
            running_loss += loss.item()

    mean_loss_by_batches = running_loss / len(model.test_loader)
    return mean_loss_by_batches


def run(
    model: PlayabilityBase,
    name: str = None,
    max_epochs: int = 100,
    lr: float = 1e-3,
    overfit=False,
):
    # Defining the name of the experiment
    timestamp = str(time()).replace(".", "")
    if name is None:
        name = f"{timestamp}_playability_net"
    else:
        name = f"{timestamp}_{name}"

    writer = SummaryWriter(log_dir=f"./runs/{name}")

    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    for epoch in range(max_epochs):
        train_loss = fit(model, optimizer)
        test_loss = test(model, epoch)
        if test_loss < best_loss:
            best_loss = test_loss
            n_without_improvement = 0

            t.save(model.state_dict(), f"./models/playability_nets/{name}_final.pt")
        else:
            if not overfit:
                n_without_improvement += 1

        print(f"Epoch {epoch}. Loss in test: {test_loss}. Loss in train: {train_loss}")
        model.report(writer, train_loss, test_loss, epoch)

        if n_without_improvement > 20:
            print(f"Stopping early. Best loss: {best_loss}")
            break

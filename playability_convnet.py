import json
from typing import List
from itertools import product
from time import time

import pandas as pd
import numpy as np
import torch as t
from torch.distributions import Bernoulli
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from shapeguard import ShapeGuard
from tqdm import tqdm


def get_level_datasets(random_state=0) -> List[TensorDataset]:
    """
    Returns train and test datasets.
    """

    # Load levels and playabilities
    df = pd.read_csv(
        "./data/array_simulation_results/samples_for_playability.csv", index_col=0
    )
    mean_p_per_l = df.groupby(["level"])["marioStatus"].mean()

    levels = []
    playabilities = []
    for l, p in mean_p_per_l.iteritems():
        levels.append(json.loads(l))
        playabilities.append(p)

    levels = np.array(levels)
    playabilities = np.array(playabilities)

    b, h, w = levels.shape
    levels_onehot = np.zeros((b, 11, h, w))
    for batch, level in enumerate(levels):
        for i, j in product(range(h), range(w)):
            c = level[i, j]
            levels_onehot[batch, c, i, j] = 1.0

    l_train, l_test, p_train, p_test = train_test_split(
        levels_onehot, playabilities, random_state=random_state
    )

    train_dataset = TensorDataset(
        t.from_numpy(l_train).type(t.float), t.from_numpy(p_train).type(t.float)
    )
    test_dataset = TensorDataset(
        t.from_numpy(l_test).type(t.float), t.from_numpy(p_test).type(t.float)
    )

    return train_dataset, test_dataset


class PlayabilityConvnet(nn.Module):
    def __init__(self, batch_size: int = 64, random_state: int = 0):
        """
        A convolutional neural network used to predict playability of
        SMB levels. Adapted from the code I wrote for Rasmus' paper.
        """
        super(PlayabilityConvnet, self).__init__()
        self.w = 14
        self.h = 14
        self.n_classes = 11
        self.input_dim = 14 * 14 * 11  # for flattening
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")

        # This assumes that the data comes as 11x14x14.
        self.logits = nn.Sequential(
            nn.Conv2d(11, 8, 5),  # output here is (8, 14-5+1, 14-5+1) = (8, 10, 10)
            nn.Tanh(),
            nn.Conv2d(8, 3, 5),  # output here is (3, 6, 6)
            nn.Tanh(),
            nn.Conv2d(3, 1, 2),  # output here is (1, 5, 5)
            nn.Tanh(),
            nn.Flatten(start_dim=1),
            nn.Linear(5 * 5, 1),
        )

        train_dataset, test_dataset = get_level_datasets(random_state=self.random_state)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        self.to(self.device)

    def forward(self, x: t.Tensor) -> Bernoulli:
        # Returns p(y | x) = Bernoulli(x; self.logits(x))
        ShapeGuard.reset()
        x.sg(("B", 11, "h", "w"))
        x = x.to(self.device)
        logits = self.logits(x)

        return Bernoulli(logits=logits)

    def binary_cross_entropy_loss(
        self, y: t.Tensor, p_y_given_x: Bernoulli
    ) -> t.Tensor:
        pred_loss = -p_y_given_x.log_prob(y)
        return pred_loss.mean()

    def report(self, writer: SummaryWriter, pred_loss, batch_id):
        writer.add_scalar("Mean Prediction Loss", pred_loss, batch_id)

        # TODO: Get some random predictions.


def fit(model: PlayabilityConvnet, optimizer: t.optim.Optimizer):
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


def test(model: PlayabilityConvnet, epoch: int):
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
    print(f"Epoch {epoch}. Loss in test: {mean_loss_by_batches}")
    return mean_loss_by_batches


def run(
    model: PlayabilityConvnet, max_epochs: int = 100, lr: float = 1e-3, overfit=False
):
    # Defining the name of the experiment
    timestamp = str(time()).replace(".", "")
    comment = f"{timestamp}_playability_convnet"

    writer = SummaryWriter(log_dir=f"./runs/{comment}")

    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    for epoch in range(max_epochs):
        train_loss = fit(model, optimizer)
        test_loss = test(model, epoch)
        if test_loss < best_loss:
            best_loss = test_loss
            n_without_improvement = 0

            t.save(model.state_dict(), f"./models/playability_convnet/model_final.pt")
        else:
            if not overfit:
                n_without_improvement += 1

        model.report(writer, train_loss, epoch)

        if n_without_improvement > 20:
            print(f"Stopping early. Best loss: {best_loss}")
            break


if __name__ == "__main__":
    pc = PlayabilityConvnet(batch_size=1)
    run(pc)

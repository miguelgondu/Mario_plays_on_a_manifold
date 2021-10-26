import json
from typing import List

import pandas as pd
import numpy as np
import torch as t
from torch.distributions import Bernoulli
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from shapeguard import ShapeGuard


def get_level_datasets() -> List[TensorDataset]:
    """
    Returns train and test datasets.
    """

    # Load levels and playabilities
    df = pd.read_csv(
        "./data/array_simulation_results/ten_random_levels.csv", index_col=0
    )
    mean_p_per_l = df.groupby(["level"])["marioStatus"].mean()

    levels = []
    playabilities = []
    for l, p in mean_p_per_l.iteritems():
        levels.append(json.loads(l))
        playabilities.append(p)

    levels = np.array(levels)
    playabilities = np.array(playabilities)

    l_train, l_test, p_train, p_test = train_test_split(levels, playabilities)

    train_dataset = TensorDataset(l_train, p_train)
    test_dataset = TensorDataset(l_test, p_test)

    return train_dataset, test_dataset


class PlayabilityConvnet(nn.Module):
    def __init__(
        self,
    ):
        """
        A convolutional neural network used to predict playability of
        SMB levels. Adapted from the code I wrote for Rasmus' paper.
        """
        super(PlayabilityConvnet, self).__init__()
        self.w = 14
        self.h = 14
        self.n_classes = 11
        self.input_dim = 14 * 14 * 11  # for flattening
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
        writer.add_scalar("Mean Prediction Loss", pred_loss.item(), batch_id)

        # TODO: Get some random predictions.


if __name__ == "__main__":
    train_dataset, test_dataset = get_level_datasets()

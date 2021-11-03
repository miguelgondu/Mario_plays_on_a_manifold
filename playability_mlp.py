import torch as t
from torch.distributions import Bernoulli
import torch.nn as nn

from playability_base import PlayabilityBase, run
from shapeguard import ShapeGuard


class PlayabilityMLP(PlayabilityBase):
    def __init__(self, batch_size: int = 64, random_state: int = 0):
        """
        An MLP used to predict playability of SMB levels.
        Adapted from the code I wrote for Rasmus' paper.
        """
        super(PlayabilityMLP, self).__init__(
            batch_size=batch_size, random_state=random_state
        )

        # This assumes that the data comes as 11x14x14.
        self.logits = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 258),
            nn.ReLU(),
            nn.Linear(258, 1),
        )
        self.to(self.device)

    def forward(self, x: t.Tensor) -> Bernoulli:
        # Returns p(y | x) = Bernoulli(x; self.logits(x))
        ShapeGuard.reset()
        x.sg(("B", 11, "h", "w"))
        x = x.view(-1, self.input_dim).sg(("B", self.input_dim))
        x = x.to(self.device)
        logits = self.logits(x)

        return Bernoulli(logits=logits)


if __name__ == "__main__":
    pc = PlayabilityMLP(batch_size=128)
    run(pc, name="mlp_balanced_loss_bs_128_w_data_augmentation")

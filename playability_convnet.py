import torch as t
from torch.distributions import Bernoulli
import torch.nn as nn

from playability_base import PlayabilityBase, run
from shapeguard import ShapeGuard


class PlayabilityConvnet(PlayabilityBase):
    def __init__(self, batch_size: int = 64, random_state: int = 0):
        """
        A convolutional neural network used to predict playability of
        SMB levels. Adapted from the code I wrote for Rasmus' paper.
        """
        super(PlayabilityConvnet, self).__init__(
            batch_size=batch_size, random_state=random_state
        )
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


if __name__ == "__main__":
    pc = PlayabilityConvnet(batch_size=64)
    run(pc, name="convnet")

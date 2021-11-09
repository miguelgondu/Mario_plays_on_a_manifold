import torch as t
from torch.distributions import Bernoulli
import torch.nn as nn

from playability_base import PlayabilityBase, run
from shapeguard import ShapeGuard


class PlayabilityConvnet(PlayabilityBase):
    def __init__(
        self, batch_size: int = 64, random_state: int = 0, augment: bool = True
    ):
        """
        A convolutional neural network used to predict playability of
        SMB levels. Adapted from the code I wrote for Rasmus' paper.

        self.l1 = t.nn.Conv2d(11, 10, 7)
        self.l2 = t.nn.Conv2d(10, 8, 4)
        self.l3 = t.nn.Conv2d(8, 4, 4)
        self.l4 = t.nn.Sequential(
            t.nn.Conv2d(4, 1, 2),
            t.nn.Flatten(),
        )
        """
        super(PlayabilityConvnet, self).__init__(
            batch_size=batch_size, random_state=random_state, augment=augment
        )
        # This assumes that the data comes as 11x14x14.
        self.logits = nn.Sequential(
            nn.Conv2d(11, 10, 7),
            nn.ReLU(),
            nn.Conv2d(10, 8, 4),
            nn.ReLU(),
            nn.Conv2d(8, 4, 4),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(16, 1),
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
    pc = PlayabilityConvnet(batch_size=128)
    # run(pc, name="convnet_w_data_augmentation_w_validation_from_dist")
    run(pc, name="balanced_loss_data_augmented_no_maxpooling")

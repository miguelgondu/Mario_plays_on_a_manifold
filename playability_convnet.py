import torch as t
from torch.distributions import Bernoulli
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from shapeguard import ShapeGuard


def preprocess_levels():
    # TODO: implement this: use load_data(), match that with playability results
    # and return datasets that iterate over levels and their playability.

    # Or even better, load up the results themselves.
    # Oops, I should get the data as we discussed with Søren: append the levels with random samples from the hierarchical network.
    pass


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

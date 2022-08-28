import torch

import gpytorch
from gpytorch.models import ExactGP

from botorch.acquisition import ExpectedImprovement
from botorch.models.model import Model


class ConstrainedExpectedImprovement(ExpectedImprovement):
    def __init__(self, model: Model, success_model: ExactGP, best_f: float) -> None:
        self.success_model = success_model
        super().__init__(model, best_f, maximize=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        ei = super().forward(X)
        distr_of_success = self.success_model(X)

        # TODO: compute the probability of success using
        # what the distr_ is. If we learn a GPC, this should
        # be as simple as predicting (?). If we learn the latent
        # function, we need to compute Prob[g(x) >= 0] = CDF_g(0) (?)
        prob_of_success = ...

        return ei * prob_of_success

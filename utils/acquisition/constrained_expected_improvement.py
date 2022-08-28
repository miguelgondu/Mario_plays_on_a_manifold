import torch

import gpytorch
from gpytorch.models import ExactGP

from botorch.acquisition import ExpectedImprovement
from botorch.models.model import Model


class ConstrainedExpectedImprovement(ExpectedImprovement):
    def __init__(self, model: Model, success_model: ExactGP, best_f: float) -> None:
        super().__init__(model, best_f, maximize=True)
        self.success_model = success_model

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        ei = super().forward(X)

        # The success model has a Dirichlet likelihood,
        # and so we need to be somewhat careful when predicting
        # probabilities of a given input X.
        # See:
        # https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/GP_Regression_on_Classification_Labels.html
        distr_of_success = self.success_model.likelihood(self.success_model(X))  # (?)

        # TODO: compute the probability of success using
        # what the distr_ is. If we learn a GPC, this should
        # be as simple as predicting (?). If we learn the latent
        # function, we need to compute Prob[g(x) >= 0] = CDF_g(0) (?)
        prob_of_success = distr_of_success.mean.flatten()

        return ei * prob_of_success

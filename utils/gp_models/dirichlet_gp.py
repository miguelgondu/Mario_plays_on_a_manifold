"""
A dirichlet GP used for classification.
"""
import torch

import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import DirichletClassificationLikelihood


class DirichletGPModel(ExactGP):
    """
    Binary classification using a Dirichlet likelihood
    function over a latent GP.
    """

    def __init__(self, train_x, train_y):
        likelihood = DirichletClassificationLikelihood(train_y)
        transformed_train_y = likelihood.transformed_targets
        super(DirichletGPModel, self).__init__(train_x, transformed_train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((1,)))
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size((1,))), batch_shape=torch.Size((1,))
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

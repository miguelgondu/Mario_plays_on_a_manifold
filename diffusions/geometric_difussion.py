import torch as t
from torch.distributions import MultivariateNormal


class GeometricDifussion:
    def __init__(self, n_points: int) -> None:
        self.n_points = n_points

    def run(self, vae) -> t.Tensor:
        """Returns the random walk as a Tensor of shape [n_points, z_dim=2]"""

        # Random starting point
        z_n = vae.embeddings[t.randint(len(vae.embeddings))]
        zs = [z_n]

        # Taking it from there.
        for _ in range(self.n_steps):
            Mz = vae.metric(z_n)

            d = MultivariateNormal(z_n, covariance_matrix=Mz.inverse())
            z_n = d.rsample()
            zs.append(z_n)

        return t.vstack(zs)

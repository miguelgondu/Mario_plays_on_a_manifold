import torch
from torch.distributions import Categorical, Normal
from vae_mario_hierarchical import VAEMarioHierarchical
from vae_geometry_base import VAEGeometryBase

Tensor = torch.Tensor


class VAEGeometryHierarchical(VAEGeometryBase, VAEMarioHierarchical):
    def __init__(
        self,
        w: int = 14,
        h: int = 14,
        z_dim: int = 2,
        n_sprites: int = 11,
        device: str = None,
    ):
        super().__init__(w, h, z_dim, n_sprites=n_sprites, device=device)

    def reweight(self, z: Tensor) -> Categorical:
        similarity = self.translated_sigmoid(self.min_distance(z)).unsqueeze(-1)
        intermediate_normal = self._intermediate_distribution(z)
        dec_mu, dec_std = intermediate_normal.mean, intermediate_normal.scale

        reweighted_std = (1 - similarity) * dec_std + similarity * (
            10.0 * torch.ones_like(dec_std)
        )
        reweighted_normal = Normal(dec_mu, reweighted_std)
        samples = reweighted_normal.rsample()
        p_x_given_z = Categorical(
            logits=samples.reshape(-1, self.h, self.w, self.n_sprites)
        )

        return p_x_given_z

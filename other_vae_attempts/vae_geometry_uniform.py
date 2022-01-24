import torch
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from vae_geometry_base import VAEGeometryBase

Tensor = torch.Tensor


class VAEGeometryUniform(VAEGeometryBase):
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
        """
        Extrapolates to uniform noise.
        """
        similarity = self.translated_sigmoid(self.min_distance(z)).view(-1, 1, 1, 1)
        dec_categorical = self.decode(z)
        dec_probs = dec_categorical.probs

        uniform_probs = (1 / self.n_sprites) * torch.ones_like(dec_probs)

        reweighted_probs = (1 - similarity) * dec_probs + similarity * (uniform_probs)
        p_x_given_z = Categorical(probs=reweighted_probs)

        return p_x_given_z


if __name__ == "__main__":
    model_name = "16324019946774652_mariovae_zdim_2_epoch_40"
    vae = VAEGeometryUniform()
    vae.load_state_dict(torch.load(f"./models/{model_name}.pt"))
    vae.update_cluster_centers(beta=-1.5, n_clusters=500)
    _, ax = plt.subplots(1, 1)
    vae.plot_latent_space(ax=ax)
    plt.show()

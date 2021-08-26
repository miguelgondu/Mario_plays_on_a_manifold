from typing import List
from geoml.discretized_manifold import DiscretizedManifold

import torch
import pandas as pd

from vae_geometry import VAEGeometry
from base_interpolation import BaseInterpolation

Tensor = torch.Tensor

# def get_playable_points(model_name):
#     df = pd.read_csv(
#         f"./data/processed/playability_experiment/{model_name}_playability_experiment.csv",
#         index_col=0,
#     )
#     playable_points = df.loc[df["marioStatus"] > 0, ["z1", "z2"]]
#     playable_points.drop_duplicates(inplace=True)
#     playable_points = playable_points.values

#     return playable_points


class GeodesicInterpolation(BaseInterpolation):
    def __init__(
        self,
        dm: DiscretizedManifold,
        model_name: str,
        n_points: int = 10,
    ):
        super().__init__(n_points=n_points)
        self.model_name = model_name
        self.dm = dm

    def interpolate(self, z: Tensor, z_prime: Tensor) -> Tensor:
        c = self.dm.connecting_geodesic(z, z_prime)
        t = torch.linspace(0, 1, self.n_points)
        interpolation = c(t)
        return interpolation

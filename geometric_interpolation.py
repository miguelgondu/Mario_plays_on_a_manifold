from typing import List
from geoml.discretized_manifold import DiscretizedManifold

import torch
import pandas as pd

from vae_geometry import VAEGeometry
from base_interpolation import BaseInterpolation

Tensor = torch.Tensor


def get_playable_points(model_name):
    df = pd.read_csv(
        f"./data/processed/playability_experiment/{model_name}_playability_experiment.csv",
        index_col=0,
    )
    playable_points = df.loc[df["marioStatus"] > 0, ["z1", "z2"]]
    playable_points.drop_duplicates(inplace=True)
    playable_points = playable_points.values

    return playable_points


class GeometricInterpolation(BaseInterpolation):
    def __init__(
        self,
        dm: DiscretizedManifold,
        model_name: str,
        n_points: int = 10,
        beta: float = -4.5,
    ):
        super().__init__(n_points=n_points)
        self.model_name = model_name
        self.beta = beta

        playable_points = get_playable_points(model_name)
        playable_points = torch.from_numpy(playable_points)
        self.playable_points = playable_points

        self.dm = dm

    def interpolate(self, z: Tensor, z_prime: Tensor) -> List[Tensor]:
        c = self.dm.connecting_geodesic(z, z_prime)
        t = torch.linspace(0, 1, self.n_points)
        interpolation = c(t)
        return interpolation

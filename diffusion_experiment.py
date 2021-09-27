import click
import torch as t

from vae_geometry_dirichlet import VAEGeometryDirichlet
from vae_geometry_hierarchical import VAEGeometryHierarchical
from vae_geometry_uniform import VAEGeometryUniform

from diffusions.geometric_difussion import GeometricDifussion
from diffusions.normal_diffusion import NormalDifussion


def load_model(extrapolation, model_name):
    if extrapolation == "dirichlet":
        Model = VAEGeometryDirichlet
    elif extrapolation == "hierarchical":
        Model = VAEGeometryHierarchical
    elif extrapolation == "Uniform":
        Model = VAEGeometryUniform

    vae = Model()
    vae.load_state_dict(t.load(f"./models/{model_name}.pt", map_location="cpu"))

    vae.update_cluster_centers()


@click.command()
@click.option("--extrapolation", type=str, default="dirichlet")
def run(extrapolation):
    vae = load_model(extrapolation)
    pass

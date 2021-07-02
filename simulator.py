"""
This script will send things to
the MarioGAN.jar compiled simulator.
"""
import subprocess
import json

import torch

from vae_mario import VAEMario
from mario_utils.levels import tensor_to_sim_level, clean_level

Tensor = torch.Tensor

JARFILE_PATH = (
    "/Users/migd/Projects/mario_geometry_project/MarioVAE_simulation/MarioGAN.jar"
)


def test_level_from_decoded_tensor(
    level: Tensor,
    human_player: bool = False,
    max_time: int = 30,
    visualize: bool = False,
) -> dict:
    if len(level.shape) < 4:
        level = level.view(1, *level.shape)
    level = tensor_to_sim_level(level)[0]
    level = str(level)

    return run_level(
        level, human_player=human_player, max_time=max_time, visualize=visualize
    )


def test_level_from_int_tensor(
    level: Tensor, human_player: bool = False, max_time: int = 30
) -> dict:
    level = clean_level(level.detach().numpy())
    level = str(level)
    print(level)

    return run_level(level, human_player=human_player)


def run_level(
    level: str, human_player: bool = False, max_time: int = 30, visualize: bool = False
) -> dict:
    # Run the MarioGAN.jar file
    if human_player:
        java = subprocess.Popen(
            ["java", "-cp", JARFILE_PATH, "geometry.PlayLevel", level],
            stdout=subprocess.PIPE,
        )
    else:
        java = subprocess.Popen(
            [
                "java",
                "-cp",
                JARFILE_PATH,
                "geometry.EvalLevel",
                level,
                str(max_time),
                str(visualize).lower(),
            ],
            stdout=subprocess.PIPE,
        )

    lines = java.stdout.readlines()
    res = lines[-1]
    res = json.loads(res.decode("utf8"))
    res["level"] = level

    return res


def test_level_from_z(z: Tensor, vae: VAEMario, human_player: bool = False) -> dict:
    """
    Passes the level that z generates
    through the simulator and returns
    a dict with results.

    These results are defined in
    MarioGAN.jar <- EvaluationInfo.
    """
    # Get the level from the VAE
    res = vae.decode(z.view(1, -1))
    level = res[0]

    return test_level_from_decoded_tensor(level, human_player=human_player)


if __name__ == "__main__":
    human_player = True
    z_dim = 2
    checkpoint = 100
    model_name = f"mariovae_zdim_{z_dim}_playesting_epoch_{checkpoint}"

    print(f"Loading model {model_name}")
    vae = VAEMario(16, 16, z_dim=z_dim)
    vae.load_state_dict(torch.load(f"./models/{model_name}.pt"))
    vae.eval()

    random_z = 2 * torch.randn((1, z_dim))
    print(f"Playing {random_z[0]}")
    res = test_level_from_z(random_z[0], vae)
    print(res)

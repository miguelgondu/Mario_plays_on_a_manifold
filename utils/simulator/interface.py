import subprocess
import json
from pathlib import Path

import torch
import numpy as np
from vae_models.vae_geometry_hierarchical import VAEGeometryHierarchical

from vae_models.vae_mario_hierarchical import VAEMarioHierarchical
from utils.mario.levels import tensor_to_sim_level, clean_level

Tensor = torch.Tensor


ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
JARFILE_PATH = f"{ROOT_DIR}/utils/simulator/simulator.jar"


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
    level: Tensor,
    human_player: bool = False,
    max_time: int = 45,
    visualize: bool = False,
) -> dict:
    level = clean_level(level.cpu().detach().numpy())
    level = str(level)

    return run_level(
        level, human_player=human_player, visualize=visualize, max_time=max_time
    )


def test_level_from_int_array(
    level: np.ndarray,
    human_player: bool = False,
    max_time: int = 45,
    visualize: bool = False,
) -> dict:
    level = clean_level(level)
    level = str(level)

    return run_level(
        level, human_player=human_player, max_time=max_time, visualize=visualize
    )


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


def test_level_from_z(
    z: Tensor, vae: VAEMarioHierarchical, human_player: bool = False
) -> dict:
    """
    Passes the level that z generates
    through the simulator and returns
    a dict with results.

    These results are defined in
    MarioGAN.jar <- EvaluationInfo.
    """
    # Get the level from the VAE
    res = vae.decode(z.view(1, -1)).probs.argmax(dim=-1)
    level = res[0]

    return test_level_from_decoded_tensor(level, human_player=human_player)


def video_for_tv2(vae: VAEGeometryHierarchical):
    # zs = torch.Tensor(
    #     [
    #         [-2.5, 0.5],
    #         [2.5, 0.5],
    #         [0.5, 2.5],
    #         [0.5, -2.5],
    #         [0.5, -2.0],
    #         [1.5, 2.5],
    #         [-1.5, -2.5],
    #     ]
    # )
    torch.manual_seed(0)
    zs = 3 * torch.randn((8, 2))

    levels: torch.Tensor
    levels = vae.decode(zs).probs.argmax(dim=-1)

    level = torch.hstack([levels[i] for i in range(levels.shape[0])])
    level_for_sim = clean_level(level.detach().numpy())

    print(level.shape)
    print(level.detach().numpy())
    run_level(str(level_for_sim), human_player=True, max_time=zs.shape[0] * 45)


def testing_playability():
    """
    Tests what's the max height for a block to be playable.
    Also tests what's the max width of a playable gap.
    """
    # basic level.
    level = 2.0 * torch.ones((14, 14))
    level[-1, :] = 0.0

    # adding a column.
    # level[-5:, 7] = 0.0

    # adding a gap
    # level[-1, :6] = 9.0

    run_level(str(clean_level(level.detach().numpy())), human_player=True)

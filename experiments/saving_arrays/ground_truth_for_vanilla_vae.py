"""
This script creates and runs the arrays for
illuminating the latent space of the vanilla
VAEs. These ground truths can then be plotted
as grids and used for obstacle building.
"""

from pathlib import Path
from pandas import array
import torch
import numpy as np

from vae_models.vae_vanilla_mario import VAEMario

from utils.simulator.simulate_array import _simulate_array

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
ARRAYS_PATH = ROOT_DIR / "data" / "arrays" / "vanilla_vae"
ARRAYS_PATH.mkdir(exist_ok=True)


def save_arrays():
    """
    Saves the arrays in ./data/arrays/vanilla_vae,
    if they're not already there.
    """

    MODELS_PATH = ROOT_DIR / "trained_models" / "vanilla_vae"
    models_paths = MODELS_PATH.glob("*.pt")

    # Getting the arrays
    for path in models_paths:
        model_name = path.stem
        array_path = ARRAYS_PATH / f"{model_name}.npz"
        if not array_path.exists():
            vae = VAEMario()
            device = vae.device
            vae.load_state_dict(torch.load(path, map_location=device))
            vae.eval()

            x_lims = [-5, 5]
            y_lims = [-5, 5]

            n_grid = 50
            z1 = np.linspace(*x_lims, n_grid)
            z2 = np.linspace(*y_lims, n_grid)

            zs = torch.Tensor([[a, b] for a in reversed(z1) for b in z2])
            cat = vae.decode(zs)

            levels = cat.probs.argmax(dim=-1).cpu().detach().numpy()

            zs = zs.detach().numpy()

            print(f"zs: {zs.shape}")
            print(f"levels: {levels.shape}")
            assert zs.shape[0] == levels.shape[0]

            print(f"Saving array for {model_name}.")
            np.savez(
                array_path,
                zs=zs,
                levels=levels,
            )


def simulate_arrays():
    """
    Simulates the arrays
    """
    arrays_paths = list(ARRAYS_PATH.glob("*.npz"))
    for i, array_path in enumerate(arrays_paths):
        print(f"Simulating array {i+1}/{len(arrays_paths)}")
        _simulate_array(
            array_path=array_path,
            processes=10,
            repetitions_per_level=5,
            exp_folder="vanilla_vae",
            verbose=True,
        )


if __name__ == "__main__":
    save_arrays()
    simulate_arrays()

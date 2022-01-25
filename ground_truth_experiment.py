"""
This experiment illuminates the latent space
by trying the decoded level several times using
Baumgarten's A* agent.

It creates an array that is can then simulated by
simulate_array.py
"""
from pathlib import Path
import torch as t
import numpy as np

from vae_mario_hierarchical import VAEMarioHierarchical
from simulate_array import _simulate_array


def get_ground_truth_arrays():
    # Hyper-arguments
    argmax = True
    n_samples = 5

    # Creating the path for the arrays.
    model_paths = Path("./models/ten_vaes").glob("*.pt")
    array_path = Path("./data/arrays/ten_vaes/ground_truth")
    array_path.mkdir(exist_ok=True, parents=True)

    # Getting the arrays
    for path in model_paths:
        model_name = path.stem
        vae = VAEMarioHierarchical()
        device = vae.device
        vae.load_state_dict(t.load(path, map_location=device))
        vae.eval()

        x_lims = [-5, 5]
        y_lims = [-5, 5]

        n_grid = 50
        z1 = np.linspace(*x_lims, n_grid)
        z2 = np.linspace(*y_lims, n_grid)

        zs = t.Tensor([[a, b] for a in reversed(z1) for b in z2])
        cat = vae.decode(zs)

        if not argmax:
            levels = cat.sample((n_samples,)).detach().numpy()
            levels = levels.reshape(n_grid * n_grid * n_samples, *levels.shape[2:])
        else:
            levels = cat.probs.argmax(dim=-1).cpu().detach().numpy()

        zs = zs.detach().numpy()
        if not argmax:
            zs = np.repeat(zs, n_samples, axis=0)

        print(f"zs: {zs.shape}")
        print(f"levels: {levels.shape}")
        assert zs.shape[0] == levels.shape[0]

        print(f"Saving array for {model_name}.")
        np.savez(
            array_path / f"{model_name}.npz",
            zs=zs,
            levels=levels,
        )


def ground_truth_experiment():
    array_paths = Path("./data/arrays/ten_vaes/ground_truth").glob("*.npz")
    for path in array_paths:
        print(f"Simualting {path}.")
        _simulate_array(path, 32, 5, exp_folder="ten_vaes/ground_truth")


if __name__ == "__main__":
    get_ground_truth_arrays()
    ground_truth_experiment()

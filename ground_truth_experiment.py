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


def ground_truth_experiment():
    # Hyper-arguments
    argmax = True
    n_samples = 5
    model_names = [f"vae_mario_hierarchical_zdim_2_id_{id_}_final" for id_ in range(5)]

    # Creating the path for the arrays.
    array_path = Path("./data/arrays/five_vaes/ground_truth")
    array_path.mkdir(exist_ok=True)

    # Getting the arrays
    for model_name in model_names:
        vae = VAEMarioHierarchical(device="cpu")
        vae.load_state_dict(t.load(f"./models/{model_name}.pt", map_location="cpu"))
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
            levels = cat.probs.argmax(dim=-1).detach().numpy()

        zs = zs.detach().numpy()
        if not argmax:
            zs = np.repeat(zs, n_samples, axis=0)

        print(f"zs: {zs.shape}")
        print(f"levels: {levels.shape}")
        assert zs.shape[0] == levels.shape[0]

        print(f"Array saved for {model_name}.")
        np.savez(
            array_path / f"{model_name}_ground_truth.npz",
            zs=zs,
            levels=levels,
        )


if __name__ == "__main__":
    ground_truth_experiment()

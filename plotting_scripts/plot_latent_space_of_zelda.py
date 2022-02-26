"""
Illuminates the latent space of zelda using
the grammar check
"""
from itertools import product
from pathlib import Path

import torch as t
import numpy as np
import matplotlib.pyplot as plt

from grammar_zelda import grammar_check
from vae_zelda_hierachical import VAEZeldaHierarchical

for id_ in range(5):
    model_name = f"zelda_hierarchical_final_{id_}"
    g_path = Path(f"./data/processed/zelda/grammar_checks/{model_name}.npz")
    vae = VAEZeldaHierarchical()
    vae.load_state_dict(t.load(f"./models/zelda/zelda_hierarchical_final_{id_}.pt"))
    x_lims = (-4, 4)
    y_lims = (-4, 4)
    n_rows = n_cols = 50
    z1 = np.linspace(*x_lims, n_cols)
    z2 = np.linspace(*y_lims, n_rows)
    positions = {
        (x, y): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
    }
    zs_in_positions = t.Tensor([z for z in positions.keys()]).type(t.float)

    if g_path.exists():
        ps = np.load(g_path)["playabilities"]
    else:
        levels = vae.decode(zs_in_positions).probs.argmax(dim=-1)
        ps = [int(grammar_check(level)) for level in levels]

    grammar_img = np.zeros((n_cols, n_rows))
    for (_, pos), p in zip(positions.items(), ps):
        grammar_img[pos] = int(p)

    encodings = vae.encode(vae.train_data).mean.detach().numpy()
    np.savez(
        f"./data/processed/zelda/grammar_checks/zelda_hierarchical_final_{id_}.npz",
        zs=zs_in_positions.detach().numpy(),
        playabilities=np.array(ps).astype(float),
    )

    _, ax = plt.subplots()
    ax.imshow(grammar_img, extent=[*x_lims, *y_lims], cmap="Blues")
    ax.scatter(encodings[:, 0], encodings[:, 1])
    ax.set_title(f"vae zelda {id_}")
    plt.show()
    plt.close()

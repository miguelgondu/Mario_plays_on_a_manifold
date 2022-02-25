"""
Illuminates the latent space of zelda using
the grammar check
"""
from itertools import product

import torch as t
import numpy as np
import matplotlib.pyplot as plt

from grammar_zelda import grammar_check
from vae_zelda_hierachical import VAEZeldaHierarchical

vae = VAEZeldaHierarchical()
vae.load_state_dict(t.load("./models/zelda/zelda_hierarchical_final.pt"))

x_lims = (-4, 4)
y_lims = (-4, 4)
n_rows = n_cols = 50
z1 = np.linspace(*x_lims, n_cols)
z2 = np.linspace(*y_lims, n_rows)

# zs = np.array([[a, b] for a, b in product(z1, z2)])
# images_dist = vae.decode(t.from_numpy(zs).type(t.float))
# images = images_dist.probs.argmax(dim=-1)
# for level in images:
#     print(grammar_check(level.detach().numpy()))

positions = {
    (x, y): (i, j) for j, x in enumerate(z1) for i, y in enumerate(reversed(z2))
}
zs_in_positions = t.Tensor([z for z in positions.keys()]).type(t.float)
levels = vae.decode(zs_in_positions).probs.argmax(dim=-1)

grammar_img = np.zeros((n_cols, n_rows))
for (_, pos), level in zip(positions.items(), levels):
    # z_ = t.Tensor(z)
    # level = vae.decode(z_).probs.argmax(dim=-1)
    p = grammar_check(level)
    print(p)
    print(level)
    print()
    grammar_img[pos] = int(p)

_, ax = plt.subplots()
ax.imshow(grammar_img, extent=[*x_lims, *y_lims], cmap="Blues")
plt.show()

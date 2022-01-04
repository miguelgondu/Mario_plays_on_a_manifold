"""
One thing that might be happening in the geometric diffusions
is that, whenever we hit a point in the boundary, it gets
stuck because inv(Sigma) is almost 0 (since the distribution
is very wide).

In this script we load the results for a 2D geometric diffusion
at random, and we analyse whether we are getting stuck on some levels.
"""
from pathlib import Path
import json
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mario_utils.plotting import get_img_from_level

geometric_diffusions = list(
    Path("./data/array_simulation_results/geometric").glob(
        "*_zdim_2_*_beta_-30_geometric_diffusion_*.csv"
    )
)
print(len(geometric_diffusions))

# Choosing one experiment at random
# p = random.choice(geometric_diffusions)
# print(f"Analyzing {p}.")

fig, axes = plt.subplots(1, 20, figsize=(10 * 7, 5))
for ax, p in zip(axes.flatten(), geometric_diffusions):
    df = pd.read_csv(p, index_col=0)
    by_z = df.groupby("z")
    playabilities = by_z["marioStatus"].mean()
    assert len(playabilities) == 11
    levels = by_z["level"].unique().values

    # Constructing a dict[level, playability]
    levels_and_playabilities = [(l[0], p) for p, l in zip(playabilities, levels)]
    left_img = np.vstack(
        [get_img_from_level(json.loads(l)) for l, _ in levels_and_playabilities]
    )

    ax.imshow(left_img)
    ax.axis("off")

fig.tight_layout()
fig.savefig("./data/plots/are_we_getting_stuck/diffusions.png")
plt.show()

# df = pd.read_csv()

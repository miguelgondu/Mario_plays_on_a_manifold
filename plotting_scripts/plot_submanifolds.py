from pathlib import Path

import matplotlib.pyplot as plt

from experiment_utils import load_csv_as_map, intersection
from geometry import DiscreteGeometry

path_to_gt = Path(
    "./data/array_simulation_results/ten_vaes/ground_truth/vae_mario_hierarchical_id_0.csv"
)
vae_path = Path("models/ten_vaes/vae_mario_hierarchical_id_0.pt")
playable_map = load_csv_as_map(path_to_gt)
strict_playability = {z: 1.0 if p == 1.0 else 0.0 for z, p in playable_map.items()}
jump_map = load_csv_as_map(path_to_gt, column="jumpActionsPerformed")
strict_jump_map = {
    z: 1.0 if jumps > 0.0 else 0.0 for z, jumps in jump_map.items()
}
p_map = intersection(strict_playability, strict_jump_map)

dg = DiscreteGeometry(p_map, "discrete_for_plotting", vae_path)
img = dg.grid
_, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.imshow(img, extent=[-5, 5, -5, 5], cmap="Blues")
ax.axis("off")
plt.show()

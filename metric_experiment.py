import torch as t
import numpy as np
import matplotlib.pyplot as plt

from vae_geometry_hierarchical import VAEGeometryHierarchical

from metric_approximation import MetricApproximation

model_name = "mario_vae_hierarchical_zdim_2_epoch_160"

vae = VAEGeometryHierarchical()
vae.load_state_dict(t.load(f"models/{model_name}.pt"))
vae.update_cluster_centers(
    model_name,
    False,
    beta=-1.5,
)
print("Updated cluster centers.")

_, (ax1, ax2) = plt.subplots(1, 2)
vae.plot_w_geodesics(ax=ax1)
# plt.show()

M = MetricApproximation(model=vae, z_dim=2, eps=0.05)
# z = t.Tensor([0.0, 0.0])

n_x, n_y = 50, 50
x_lims = (-6, 6)
y_lims = (-6, 6)
z1 = t.linspace(*x_lims, n_x)
z2 = t.linspace(*y_lims, n_x)
positions = {
    (x.item(), y.item()): (i, j)
    for j, x in enumerate(z1)
    for i, y in enumerate(reversed(z2))
}

zs = t.Tensor([[x, y] for x in z1 for y in z2])
metric_volume = np.zeros((n_y, n_x))
for l, z in enumerate(zs):
    (x, y) = z
    i, j = positions[(x.item(), y.item())]
    Mz = M(z)
    # print(Mz)
    detMz = t.det(Mz).item()
    if detMz < 0:
        metric_volume[i, j] = np.nan
    else:
        metric_volume[i, j] = np.log(detMz)

cbar = ax2.imshow(metric_volume, extent=[*x_lims, *y_lims], cmap="Blues")
plt.colorbar(cbar)
plt.tight_layout()
plt.savefig("./data/plots/metric_volume.png")
plt.show()

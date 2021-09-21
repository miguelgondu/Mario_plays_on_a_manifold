import torch as t
import numpy as np
import matplotlib.pyplot as plt

from vae_geometry_hierarchical import VAEGeometryHierarchical

from metric_approximation import MetricApproximation

model_name = "mario_vae_hierarchical_zdim_2_epoch_160"

vae = VAEGeometryHierarchical()
vae.load_state_dict(t.load(f"models/{model_name}.pt"))
print("Updating cluster centers")
vae.update_cluster_centers(
    model_name,
    False,
    beta=-1.5,
)

_, ax = plt.subplots(1, 1)
vae.plot_w_geodesics(ax=ax)
plt.show()

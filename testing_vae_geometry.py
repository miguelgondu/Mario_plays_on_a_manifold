import torch
import matplotlib.pyplot as plt

from vae_geometry import VAEGeometry

model_name = "mariovae_z_dim_2_overfitting_epoch_100"
vae = VAEGeometry()
vae.load_state_dict(torch.load(f"models/{model_name}.pt"))
vae.update_cluster_centers(model_name, False, beta=-1.5)
print(vae)

# vae.plot_latent_space()
vae.plot_w_geodesics()
plt.show()

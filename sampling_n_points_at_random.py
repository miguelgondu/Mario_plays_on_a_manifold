"""
This script samples the latent space of
some models (varying in latent space shape),
and saves an array with their levels for simulation.
"""
import torch as t
import numpy as np

from vae_mario_hierarchical import VAEMarioHierarchical

models = {
    2: "16388917374131331_mariovae_zdim_2_normal_final",
    8: "1638894528256156_mariovae_zdim_8_normal_final",
    32: "16388927503019269_mariovae_zdim_32_normal_final",
    64: "16388929591033669_mariovae_zdim_64_normal_final",
}

for z_dim, model_name in models.items():
    vae = VAEMarioHierarchical(z_dim=z_dim)
    vae.load_state_dict(t.load(f"./models/{model_name}.pt"))
    vae.eval()

    zs = vae.p_z.sample((1000,))
    levels = vae.decode(zs).probs.argmax(dim=-1)

    np.savez(
        f"./data/arrays/{model_name}.npz",
        zs=zs.detach().numpy(),
        levels=levels.detach().numpy(),
    )

    print(f"Saved array for model {model_name}.")

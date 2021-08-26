import torch
import matplotlib.pyplot as plt

from mario_utils.plotting import plot_level_from_decoded_tensor

from vae_geometry import VAEGeometry

model_name = "mariovae_z_dim_16_epoch_280"
for beta in [10.0, 5.0, 2.5, 0.01, -0.01, -1.0]:
    vae = VAEGeometry(z_dim=16)
    vae.load_state_dict(torch.load(f"models/{model_name}.pt"))
    vae.update_cluster_centers(model_name, False, n_clusters=50, beta=beta)

    print(vae)
    print(vae.cluster_centers)
    print(vae.cluster_centers.shape)

    zs = torch.randn((5, 16))
    og_levels = vae.decode(zs)
    reweighted_levels = vae.reweight(zs)[0]

    similarity = vae.translated_sigmoid(vae.min_distance(zs)).unsqueeze(-1)
    print(similarity)

    fig, (axes_og, axes_reweighted) = plt.subplots(2, 5, figsize=(5 * 7, 7))

    for og_level, ax in zip(og_levels, axes_og):
        plot_level_from_decoded_tensor(og_level.unsqueeze(0), ax)

    for reweighted_level, ax in zip(reweighted_levels, axes_reweighted):
        plot_level_from_decoded_tensor(reweighted_level.unsqueeze(0), ax)

    fig.suptitle(f"beta = {beta}")
    plt.tight_layout()

    plt.savefig(f"./data/plots/multiple_betas/beta_{str(beta).replace('.', '_')}.png")
    plt.close()
    # plt.show()

import torch as t
import matplotlib.pyplot as plt

from vae_text import VAEText

vaetext = VAEText()
vaetext.load_state_dict(t.load(f"./models/test_text_final.pt"))

semantic = vaetext.plot_correctness("semantic")
syntactic = vaetext.plot_correctness("syntactic")

_, (ax1, ax2) = plt.subplots(1, 2)

ax1.imshow(syntactic, extent=[-5, 5, -5, 5], cmap="Blues")
latent_codes, _ = vaetext.forward(vaetext.train_tensor)
latent_codes = latent_codes.mean.detach().numpy()
ax1.scatter(latent_codes[:, 0], latent_codes[:, 1], c="k", marker="x")

ax2.imshow(semantic, extent=[-5, 5, -5, 5], cmap="Blues")
ax2.scatter(latent_codes[:, 0], latent_codes[:, 1], c="k", marker="x")

plt.show()

vaetext.plot_correctness("semantic")

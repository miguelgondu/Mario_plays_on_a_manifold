import torch

from vae_mario import VAEMario
from simulator import test_level_from_decoded_tensor

model_name = "mariovae_z_dim_2_overfitting_epoch_480"
model = VAEMario(14, 14, 2)
model.load_state_dict(torch.load(f"./models/{model_name}.pt"))

z = torch.randn((1, 2))
test_level_from_decoded_tensor(model.decode(z), human_player=True)

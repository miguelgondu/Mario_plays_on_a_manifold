# Hierarchical
python diffusion_experiment.py deeper_lr_1e-4_no_overfit_final --extrapolation=hierarchical
python diffusion_experiment.py playable_lr_1e-4_no_overfit_final --extrapolation=hierarchical --only-playable

# Dirichlet
python diffusion_experiment.py vae_deeper_lr_1e-4_no_overfit_final.pt --extrapolation=dirichlet
python diffusion_experiment.py playable_vae_deeper_lr_1e-4_no_overfit_final.pt --extrapolation=dirichlet --only-playable

# Uniform
python diffusion_experiment.py vae_deeper_lr_1e-4_no_overfit_final.pt --extrapolation=uniform
python diffusion_experiment.py playable_vae_deeper_lr_1e-4_no_overfit_final.pt --extrapolation=uniform --only-playable

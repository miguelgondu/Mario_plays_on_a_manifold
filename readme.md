# Code for *Mario plays on a manifold*

This repository contains the code used in the paper [*Mario Plays on a Manifold: Generating Functional Content in Latent Space through Differential Geometry*](https://ieee-cog.org/2022/assets/papers/paper_112.pdf), in which we modify the geometry of 2-dimensional Variational Autoencoders trained on levels from *Super Mario Bros* and *The Legend of Zelda*, taken from [The Video Game Level Corpus](https://github.com/TheVGLC/TheVGLC).

## A guide through the code

We will focus on Super Mario Bros in this guide.

### Training the VAEs

After postprocessing the levels, they can be found in `data/processed/all_levels_onehot.npz` and `data/processed/all_playable_levels_onehot.npz`. These are the dataset used to train the VAEs.

The definition of the models can be found under `vae_models/*.py`.

The training scripts can be found under `experiments/train_*.py`. We trained 10 different VAEs for each game, and the weights are available upon request. (The files are too heavy for GitHub)

### Finding the playable regions

To modify the geometry, we first need to find where the playable levels are. Using the trained VAEs, we decode grids in latent space of size $50\times 50$. The scripts that save these levels are in `experiments/saving_arrays/ground_truth_experiment.py`.

### Geometries, or how to interpolate and diffuse safely

Under `geometries` you will find an implementation of our approach (`discetized_geometry.py`) for safe interpolation and sampling, alongside the baselines `normal_geometry.py` and `baseline_geometry.py`. A `geometry` is just an interface for an interpolation and a diffusion algorithm, all of which are under `interpolations` and `diffusions` respecitvely.

### Restricted Bayesian Optimization

In our final experiment, we restrict the domain of Bayesian Optimization to only the playable regions in latent space. The scripts that run this experiment are in `experiments/bayesian_optimization`.
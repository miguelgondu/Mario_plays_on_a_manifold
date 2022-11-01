from utils.experiment.bayesian_optimization import run_first_samples
from utils.experiment import load_model

if __name__ == "__main__":
    vae = load_model()
    run_first_samples(vae, force=True, name="random_samples")

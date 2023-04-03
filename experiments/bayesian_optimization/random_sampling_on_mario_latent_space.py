from pathlib import Path
from utils.experiment.bayesian_optimization import run_first_samples
from utils.experiment import load_model

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()

if __name__ == "__main__":
    model_id = 0
    vae = load_model(model_id=model_id)
    data_dir = ROOT_DIR / "data" / "bayesian_optimization" / "traces"
    for exp_id in range(10):
        save_path = data_dir / f"random_samples_{model_id}_{exp_id}.npz"
        run_first_samples(vae, force=True, save_path=save_path)

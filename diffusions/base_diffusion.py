import torch


class BaseDiffusion:
    def __init__(self, z_0, n_steps: int = 100) -> None:
        self.n_steps = n_steps

    def run(self) -> torch.Tensor:
        raise NotImplementedError

import torch as t


class BaseDiffusion:
    def __init__(self, n_steps: int = 100) -> None:
        self.n_steps = n_steps

    def run(self, initial_points: t.Tensor) -> t.Tensor:
        raise NotImplementedError

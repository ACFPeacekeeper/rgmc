import torch

from input_transformations.noise import Noise

class GaussianNoise(Noise):
    def __init__(self, mean=0., std=1.) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn(x.size()) * self.std + self.mean

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

import torch

class Noise(object):
    def __init__(self, mean=0., std=1.) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

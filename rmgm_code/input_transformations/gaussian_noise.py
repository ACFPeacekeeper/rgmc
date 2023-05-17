import torch

from input_transformations.noise import Noise

class GaussianNoise(Noise):
    def __init__(self, device, mean=0., std=1.):
        self.mean = mean
        self.std = std

        self.device = device

    def add_noise(self, x):
        return torch.clamp(x + torch.randn(x.size()).to(self.device) * self.std + self.mean, torch.min(x), torch.max(x))

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

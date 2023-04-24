import torch
import torch.nn as nn

class AdversarialAttack(object):
    def __init__(self, name: str, model: nn.Module) -> None:
        self.name = name
        self.model = model

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
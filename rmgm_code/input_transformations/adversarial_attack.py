import torch
import torch.nn as nn

class AdversarialAttack(object):
    def __init__(self, name: str, model: nn.Module) -> None:
        self.name = name
        self.model = model

    def example_generation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def __repr__(self) -> str:
        raise NotImplementedError
import timm
import torch
from torch import nn


class TimmModel(nn.Module):
    def __init__(
        self, model_name: str, num_classes: int = 1000, pretrained: bool = True, in_chans: int = 3
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, in_chans=in_chans, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

from typing import Optional

import segmentation_models_pytorch as smp
import timm
import torch
from torch import nn


class TimmModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 1000,
        pretrained: bool = True,
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SMPModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        encoder_name: str,
        in_channels: int = 3,
        num_classes: int = 1,
        encoder_weights: Optional[str] = "imagenet",
    ) -> None:
        super().__init__()
        self.model = smp.create_model(
            arch=model_name,
            encoder_name=encoder_name,
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_weights=encoder_weights,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

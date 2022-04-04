from torch import nn
import timm


class TimmModel(nn.Module):
    def __init__(self, model_name, num_classes=1000, pretrained=True, in_chans=3):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=in_chans, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
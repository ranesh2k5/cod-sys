import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class EfficientNetBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_b4(weights=weights)
        features = base.features

        self.stage1 = features[0:2]
        self.stage2 = features[2:3]
        self.stage3 = features[3:5]
        self.stage4 = features[5:7]
        self.stage5 = features[7:]

        # Detect actual output channels dynamically
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            f1 = self.stage1(dummy)
            f2 = self.stage2(f1)
            f3 = self.stage3(f2)
            f4 = self.stage4(f3)
            f5 = self.stage5(f4)
        self.out_channels = [f1.shape[1], f2.shape[1], f3.shape[1], f4.shape[1], f5.shape[1]]
        print(f"Backbone channels: {self.out_channels}")

    def forward(self, x: torch.Tensor):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)
        return f1, f2, f3, f4, f5

import torch.nn as nn
from torchvision import models


class PneumoniaDenseNet(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3, pretrained=True):
        super().__init__()
        self.backbone = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = True


def build_model(cfg):
    m = cfg["model"]
    return PneumoniaDenseNet(
        num_classes=m["num_classes"],
        dropout=m["dropout"],
    )

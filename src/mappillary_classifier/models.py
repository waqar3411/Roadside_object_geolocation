from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)


def create_resnet_classifier(
    model_name: str = "resnet34",
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    """Create a ResNet classifier with a new final layer."""
    model_name = model_name.lower()

    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)

    elif model_name == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)

    elif model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)

    elif model_name == "resnet101":
        weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet101(weights=weights)

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        hidden_channels = max(channels // reduction, 1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_channels, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.size()

        y = self.global_avg_pool(x).view(batch_size, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)

        return x * y.expand_as(x)


class SEResNet34(nn.Module):
    """ResNet34 with SE attention and multi-level feature concatenation."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        reduction: int = 16,
    ):
        super().__init__()

        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.resnet34(weights=weights)

        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.se1 = SEBlock(64, reduction)
        self.se2 = SEBlock(128, reduction)
        self.se3 = SEBlock(256, reduction)
        self.se4 = SEBlock(512, reduction)

        self.fc = nn.Linear(64 + 128 + 256 + 512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.se1(self.layer1(x))
        x2 = self.se2(self.layer2(x1))
        x3 = self.se3(self.layer3(x2))
        x4 = self.se4(self.layer4(x3))

        p1 = self.avgpool(x1).flatten(1)
        p2 = self.avgpool(x2).flatten(1)
        p3 = self.avgpool(x3).flatten(1)
        p4 = self.avgpool(x4).flatten(1)

        features = torch.cat([p1, p2, p3, p4], dim=1)
        return self.fc(features)


def create_model(
    model_name: str = "resnet34",
    num_classes: int = 2,
    pretrained: bool = True,
    se_reduction: int = 16,
) -> nn.Module:
    """Create a supported classifier."""
    model_name = model_name.lower()

    if model_name == "seresnet34":
        return SEResNet34(
            num_classes=num_classes,
            pretrained=pretrained,
            reduction=se_reduction,
        )

    return create_resnet_classifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )


def get_gradcam_target_layer(model: nn.Module, model_name: str):
    """Return the final convolutional layer for Grad-CAM."""
    if hasattr(model, "module"):
        model = model.module

    if hasattr(model, "layer4"):
        return model.layer4[-1]

    raise ValueError(f"Grad-CAM target layer is not defined for {model_name}.")

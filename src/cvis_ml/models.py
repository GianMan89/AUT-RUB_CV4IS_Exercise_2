
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


class TransferModelFactory:
    """Factory for small transfer-learning CNN baselines.

    Supported architectures:
    - resnet18: He et al., Deep Residual Learning for Image Recognition, CVPR 2016.
    - mobilenet_v3_small: Howard et al., Searching for MobileNetV3, ICCV 2019.
    - efficientnet_b0: Tan & Le, EfficientNet, ICML 2019.

    Supported strategies:
    - frozen: freeze backbone, train classifier head only.
    - partial: unfreeze a small later part of the network plus the head.
    - full: fine-tune all parameters with a small learning rate.
    """
    @staticmethod
    def create(architecture: str, num_classes: int, strategy: str = "frozen", pretrained: bool = True) -> nn.Module:
        arch = architecture.lower()
        if arch == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
            TransferModelFactory._set_trainability_resnet18(model, strategy)
            return model
        if arch == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            TransferModelFactory._set_trainability_mobilenet(model, strategy)
            return model
        if arch == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            TransferModelFactory._set_trainability_efficientnet(model, strategy)
            return model
        raise ValueError(f"Unknown architecture: {architecture}")

    @staticmethod
    def _freeze_all(model: nn.Module):
        for p in model.parameters():
            p.requires_grad = False

    @staticmethod
    def _set_trainability_resnet18(model: nn.Module, strategy: str):
        strategy = strategy.lower()
        if strategy == "frozen":
            TransferModelFactory._freeze_all(model)
            for p in model.fc.parameters():
                p.requires_grad = True
        elif strategy == "partial":
            TransferModelFactory._freeze_all(model)
            for p in model.layer4.parameters():
                p.requires_grad = True
            for p in model.fc.parameters():
                p.requires_grad = True
        elif strategy == "full":
            for p in model.parameters():
                p.requires_grad = True
        else:
            raise ValueError("strategy must be frozen, partial, or full")

    @staticmethod
    def _set_trainability_mobilenet(model: nn.Module, strategy: str):
        strategy = strategy.lower()
        if strategy == "frozen":
            TransferModelFactory._freeze_all(model)
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif strategy == "partial":
            TransferModelFactory._freeze_all(model)
            for p in model.features[-2:].parameters():
                p.requires_grad = True
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif strategy == "full":
            for p in model.parameters():
                p.requires_grad = True
        else:
            raise ValueError("strategy must be frozen, partial, or full")

    @staticmethod
    def _set_trainability_efficientnet(model: nn.Module, strategy: str):
        strategy = strategy.lower()
        if strategy == "frozen":
            TransferModelFactory._freeze_all(model)
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif strategy == "partial":
            TransferModelFactory._freeze_all(model)
            for p in model.features[-2:].parameters():
                p.requires_grad = True
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif strategy == "full":
            for p in model.parameters():
                p.requires_grad = True
        else:
            raise ValueError("strategy must be frozen, partial, or full")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def describe_trainable_parameters(model: nn.Module, max_rows: int = 12):
    """Return a small list of trainable parameter tensors for inspection."""
    rows = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            rows.append({"name": name, "shape": tuple(p.shape), "numel": p.numel()})
        if len(rows) >= max_rows:
            break
    return rows


from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetConfig:
    """Configuration for a dataset used in the exercise."""
    name: str
    data_root: str = "./data_cache"
    image_size: int = 96
    batch_size: int = 32
    num_workers: int = 2
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    seed: int = 42


@dataclass
class ModelConfig:
    """Configuration for a transfer-learning model."""
    architecture: str = "resnet18"  # resnet18, mobilenet_v3_small, efficientnet_b0
    strategy: str = "frozen"        # frozen, partial, full
    pretrained: bool = True
    num_classes: int = 10


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "auto"
    class_weights: bool = False
    max_batches_per_epoch: Optional[int] = None


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    dataset: DatasetConfig
    model: ModelConfig
    train: TrainConfig
    experiment_name: str = "experiment"

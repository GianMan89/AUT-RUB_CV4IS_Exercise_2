
# CV4IS Exercise 2 — Transfer Learning Baselines in PyTorch

This repository contains the second exercise for **Computer Vision for Industrial Systems**.

The exercise is aligned with Lecture 3: **Machine Learning Foundations for Industrial Vision**.

The focus is not on building model architectures from scratch. Instead, students learn to work with the full practical transfer-learning pipeline:

- load a dataset,
- define preprocessing and augmentation,
- inspect image tensors and labels,
- create DataLoaders,
- adapt pretrained convolutional neural networks,
- train transfer-learning baselines,
- compare learning curves,
- inspect confusion matrices and hard cases,
- reason about preprocessing, label quality, class imbalance, and industrial validity.

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/01_guided_cifar10_transfer_learning.ipynb` | Guided first pipeline on CIFAR-10 |
| `notebooks/02_industrial_mvtec_capsule_transfer_learning.ipynb` | More independent industrial transfer-learning task |
| `notebooks/solutions/01_guided_cifar10_transfer_learning_solution.ipynb` | Teacher solution |
| `notebooks/solutions/02_industrial_mvtec_capsule_transfer_learning_solution.ipynb` | Teacher solution |

## Datasets

1. **CIFAR-10** through `torchvision.datasets.CIFAR10` for the first guided task.
2. **MVTec Capsule subset** through Hugging Face dataset `alexsu52/mvtec_capsule` for the industrial task.

The MVTec exercise uses a simplified supervised teaching split. This is not the official MVTec anomaly-detection protocol.

## Binder

Use `.binder/requirements.txt`. The first launch will download PyTorch, TorchVision, pretrained weights, and datasets. This can take time.

For in-class use, start Binder early or pre-cache the environment.

## Recommended teaching use

- 20–30 min: instructor explains CNNs, transfer learning, preprocessing, and the exercise workflow.
- 60–70 min: students work mainly on notebook 1, then start notebook 2 if they are fast.

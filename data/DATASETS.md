# Dataset notes

## CIFAR-10

Used in Notebook 1.

- Source: TorchVision
- Role: guided, non-industrial first transfer-learning exercise
- Reason: small, RGB, stable, directly loadable through PyTorch/TorchVision

## MVTec Capsule teaching split

Used in Notebook 2.

- Source: Hugging Face dataset `alexsu52/mvtec_capsule`
- Role: industrially motivated binary classification task
- Labels: normal / abnormal
- Important caveat: this is a supervised teaching split, not the official MVTec AD anomaly-detection benchmark protocol.

## Why not the full MVTec AD dataset here?

The full MVTec AD dataset is excellent for the semester project, but it is too large and too protocol-specific for this first transfer-learning exercise. This exercise focuses on the pipeline and experimental thinking, not on official benchmark performance.

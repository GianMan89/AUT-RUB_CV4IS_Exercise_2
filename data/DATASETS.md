
# Dataset notes

## CIFAR-10

Used for the first guided notebook. It is loaded through `torchvision.datasets.CIFAR10`.

CIFAR-10 is not industrial, but it is a convenient RGB classification dataset for learning the transfer-learning pipeline.

## MVTec Capsule subset

Used for the second, industrially oriented notebook. It is loaded through Hugging Face dataset:

`alexsu52/mvtec_capsule`

This dataset is derived from the MVTec AD capsule category. Labels are normal / abnormal.

Teaching caveat:
The original MVTec AD benchmark is intended for anomaly detection with defect-free training data. In this exercise, we create a supervised teaching split from the available examples so students can practice transfer learning for binary classification. This should not be presented as the official anomaly-detection protocol.

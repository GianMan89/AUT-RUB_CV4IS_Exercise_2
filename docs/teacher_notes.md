
# Teacher notes — Exercise 2

## Learning goals

Students should be able to:

1. explain why preprocessing is part of the model contract;
2. distinguish frozen feature extraction, partial fine-tuning, and full fine-tuning;
3. adapt a pretrained CNN to a new number of classes;
4. train a small transfer-learning baseline in PyTorch;
5. read learning curves and confusion matrices;
6. interpret results in terms of data, labels, augmentation, class imbalance, and industrial requirements.

## Suggested 90-minute flow

| Time | Activity |
|---:|---|
| 0–10 min | CNN intuition: convolution, channels, feature maps, pooling |
| 10–20 min | Transfer learning strategies: frozen, partial, full |
| 20–25 min | Notebook orientation and Binder check |
| 25–55 min | Notebook 1: guided CIFAR-10 pipeline |
| 55–80 min | Notebook 2: industrial MVTec Capsule task |
| 80–90 min | Discussion: what changed, what failed, what would be industrially risky? |

## Suggested baselines

Use small settings to fit the exercise:

- CIFAR-10 subset: 1000–2000 train samples, 400–500 validation samples, 1 epoch.
- Industrial MVTec Capsule subset: 160 train samples, 80 validation samples, 1 epoch.

Students should focus on the workflow and interpretation, not on leaderboard performance.

## Hidden factors to emphasize

- Transfer learning does not remove the need for valid preprocessing.
- Train/validation splits must respect data generation, batches, parts, and acquisition conditions.
- Augmentations must be operationally plausible.
- Accuracy can be misleading under imbalance.
- Good baselines are diagnostic experiments, not final products.

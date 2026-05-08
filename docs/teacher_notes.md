# Teacher notes — Exercise 2

## Session design

The exercise is designed for a 90-minute session:

| Time | Activity |
|---:|---|
| 0–25 min | Short instructor introduction to CNNs and transfer learning |
| 25–35 min | Students open Binder and inspect CIFAR-10 notebook |
| 35–60 min | Guided CIFAR-10 baseline + first experiment |
| 60–80 min | Industrial MVTec Capsule notebook |
| 80–90 min | Discussion: what changed performance and why? |

## Why training is longer now

One epoch is not enough for students to see stable learning behavior. The default settings are now:

- CIFAR-10 frozen ResNet18: 3 epochs, with optional 4–5 epoch comparisons.
- MVTec Capsule: 5 epochs baseline, with optional 8 epoch comparison.

If the class is CPU-only and training is slow, students can use `max_batches_per_epoch`, but the default should train through all selected data.

## Pin memory warning fix

The data modules now set:

```python
pin_memory=torch.cuda.is_available()
```

This avoids the warning:

```text
'pin_memory' argument is set as true but no accelerator is found
```

on CPU-only machines. It keeps pinned memory active only when CUDA is available.

## Suggested instructor emphasis

Students should not only run cells. They should answer:

- What did changing epochs do?
- What did changing learning rate do?
- What did augmentation do?
- What did transfer strategy change?
- Which model is best if runtime matters?
- Which error type is more critical for industrial inspection?

## Common student difficulties

- Confusing “pretrained” with “already solved”.
- Forgetting that ImageNet normalization is part of the model contract.
- Using too high a learning rate for partial fine-tuning.
- Comparing validation results from different datasets or splits without caution.
- Treating accuracy as sufficient for the industrial binary task.
- Ignoring false normals / false abnormals.

## Optional challenge

For fast groups:

- run the same MVTec setting with and without augmentation,
- enable class weights,
- compare MobileNetV3-Small vs ResNet18,
- inspect misclassified examples and write a failure taxonomy.

# CV4IS Exercise 2 — Transfer Learning Baselines in PyTorch

This repository contains Exercise 2 for **Computer Vision for Industrial Systems**.

The exercise is aligned with the lecture on **Machine Learning Foundations for Computer Vision**. It focuses on practical transfer learning rather than building architectures from scratch.

## Main learning message

Transfer learning is not only “load a pretrained model”.

It is a complete pipeline:

1. dataset and split,
2. preprocessing and augmentation,
3. pretrained CNN backbone,
4. classifier-head adaptation,
5. transfer strategy,
6. optimizer and learning rate,
7. number of epochs,
8. validation metrics,
9. confusion matrix and visual error analysis.

## What changed in this revised version

This version addresses three teaching issues:

1. **No distracting `pin_memory` CPU warning**  
   `pin_memory=True` is now used only when CUDA is available.

2. **Longer training**  
   The notebooks no longer use one-epoch toy training as the default. Students start with 3–5 epochs and are asked to compare longer runs.

3. **More student experimentation**  
   Students now vary learning rate, epochs, architecture, transfer strategy, augmentation, and class weights. They are also asked to look up PyTorch / TorchVision documentation themselves.

## Repository structure

```text
notebooks/
  01_guided_cifar10_transfer_learning.ipynb
  02_industrial_mvtec_capsule_transfer_learning.ipynb
  solutions/
    01_guided_cifar10_transfer_learning_solution.ipynb
    02_industrial_mvtec_capsule_transfer_learning_solution.ipynb

src/cvis_ml/
  config.py
  data.py
  models.py
  engine.py
  visualization.py

docs/
  teacher_notes.md

slides/
  intro_20min_outline.md

data/
  DATASETS.md

.binder/
  requirements.txt
```

## Notebooks

### 1. Guided CIFAR-10 notebook

The first notebook is guided and uses CIFAR-10.

Students learn:

- CNN feature-map shape intuition,
- TorchVision dataset loading,
- ImageNet-style preprocessing,
- frozen ResNet18 transfer learning,
- learning-rate and epoch experiments,
- augmentation experiment,
- architecture / strategy comparison,
- confusion matrix and misclassified-example inspection.

### 2. Industrial MVTec Capsule notebook

The second notebook is more independent and uses a Hugging Face MVTec Capsule teaching split.

Students learn:

- how to configure a more industrial dataset,
- how to train a binary normal/abnormal classifier,
- why the setup is not the official anomaly-detection benchmark,
- why macro-F1 matters,
- how class imbalance changes interpretation,
- how transfer strategy, epochs, learning rate, and augmentation affect results,
- how to discuss false normal and false abnormal errors.

## Binder note

The first launch may take time because Binder must install PyTorch/TorchVision and download:

- CIFAR-10,
- pretrained model weights,
- Hugging Face MVTec Capsule dataset.

For live teaching, ask students to start Binder before the exercise begins.

## Local execution

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
jupyter lab
```

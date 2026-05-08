
# 20–30 minute intro slide outline for Exercise 2

## Slide 1 — Today: transfer learning pipeline
- load data
- preprocess images
- adapt a pretrained CNN
- train baseline
- evaluate and inspect errors

## Slide 2 — CNN intuition
- convolutional filters scan over the image
- early layers detect edges / colors / textures
- later layers combine features into object-level evidence

## Slide 3 — What is transferred?
- pretrained backbone learned from large natural-image data
- new task-specific head is trained for our labels
- optional fine-tuning adapts later features

## Slide 4 — Three strategies
- frozen feature extractor: train head only
- partial fine-tuning: unfreeze later layers
- full fine-tuning: update all parameters

## Slide 5 — Preprocessing is part of the model contract
- resize
- tensor layout
- normalization
- color handling
- augmentation
- train/inference consistency

## Slide 6 — What students do today
- CIFAR-10 guided pipeline
- industrial MVTec Capsule binary task
- compare baselines and learning curves
- interpret errors


import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay


def denormalize_imagenet(tensor):
    """Denormalize a tensor normalized with ImageNet mean/std."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device)[:, None, None]
    return torch.clamp(tensor * std + mean, 0, 1)


def show_batch(loader, class_names, n=8):
    images, labels = next(iter(loader))
    n = min(n, len(images))
    cols = min(4, n)
    rows = math.ceil(n / cols)
    plt.figure(figsize=(3 * cols, 3 * rows))
    for i in range(n):
        img = denormalize_imagenet(images[i]).permute(1, 2, 0).cpu().numpy()
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
    plt.tight_layout()


def plot_history(history, title="learning curves"):
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train loss")
    plt.plot(epochs, [h["val_loss"] for h in history], marker="o", label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title + " - loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [h["train_acc"] for h in history], marker="o", label="train acc")
    plt.plot(epochs, [h["val_acc"] for h in history], marker="o", label="val acc")
    plt.plot(epochs, [h["val_macro_f1"] for h in history], marker="o", label="val macro F1")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.title(title + " - metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def show_confusion_matrix(y_true, y_pred, class_names, title="confusion matrix"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names,
                                            xticks_rotation=45, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()

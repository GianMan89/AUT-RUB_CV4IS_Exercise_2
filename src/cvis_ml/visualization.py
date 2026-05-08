import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay


def denormalize_imagenet(tensor):
    """Denormalize a tensor normalized with ImageNet mean/std."""
    device = tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
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
    plt.figure(figsize=(11, 4))

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
    plt.ylim(0, 1.02)
    plt.title(title + " - metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()


def show_confusion_matrix(y_true, y_pred, class_names, title="confusion matrix"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        xticks_rotation=45,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    plt.tight_layout()


@torch.no_grad()
def collect_predictions(model, loader, device, max_batches=None):
    """Collect images, labels, predictions and probabilities from a loader."""
    model.eval()
    device = torch.device(device)
    rows = []
    batch_images = []
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1).cpu()
        preds = probs.argmax(dim=1)
        for i in range(len(images)):
            rows.append({
                "label": int(labels[i]),
                "pred": int(preds[i]),
                "confidence": float(probs[i, preds[i]]),
                "probabilities": probs[i].numpy(),
            })
            batch_images.append(images[i].cpu())
    return batch_images, rows


@torch.no_grad()
def show_misclassified(model, loader, class_names, device, n=8, max_batches=20):
    """Visualize a few misclassified examples."""
    images, rows = collect_predictions(model, loader, device=device, max_batches=max_batches)
    mistakes = [(img, r) for img, r in zip(images, rows) if r["label"] != r["pred"]]
    if len(mistakes) == 0:
        print("No misclassified examples found in the inspected batches.")
        return

    mistakes = sorted(mistakes, key=lambda x: x[1]["confidence"], reverse=True)[:n]
    cols = min(4, len(mistakes))
    rows_n = math.ceil(len(mistakes) / cols)
    plt.figure(figsize=(3.2 * cols, 3.4 * rows_n))
    for i, (img_tensor, r) in enumerate(mistakes):
        img = denormalize_imagenet(img_tensor).permute(1, 2, 0).numpy()
        plt.subplot(rows_n, cols, i + 1)
        plt.imshow(img)
        plt.title(
            f"true: {class_names[r['label']]}\n"
            f"pred: {class_names[r['pred']]}\n"
            f"conf: {r['confidence']:.2f}",
            fontsize=9,
        )
        plt.axis("off")
    plt.tight_layout()


def results_table(results):
    """Create a compact pandas DataFrame from ExperimentResult objects."""
    import pandas as pd
    rows = []
    for r in results:
        rows.append({
            "name": r.name,
            "val_acc": r.metrics["accuracy"],
            "val_macro_f1": r.metrics["macro_f1"],
            "val_loss": r.metrics["loss"],
            "runtime_s": r.metrics["runtime_s"],
        })
    return pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)

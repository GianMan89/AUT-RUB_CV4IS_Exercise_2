from dataclasses import dataclass
from typing import Dict, List, Optional
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm.auto import tqdm


@dataclass
class ExperimentResult:
    name: str
    history: List[Dict]
    metrics: Dict
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray


def resolve_device(device="auto"):
    """Resolve device string for teaching notebooks."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS support can be useful on Apple Silicon, but is optional.
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class Trainer:
    """Small training/evaluation engine for image classification."""
    def __init__(
        self,
        model,
        device="auto",
        class_weights=None,
        learning_rate=1e-3,
        weight_decay=1e-4,
        label_smoothing=0.0,
        scheduler=None,
        epochs_for_scheduler=1,
    ):
        self.device = resolve_device(device)
        self.model = model.to(self.device)

        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        else:
            weight_tensor = None

        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=float(label_smoothing))
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found. Check the transfer-learning strategy.")

        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

        self.scheduler = None
        if scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, int(epochs_for_scheduler)),
            )

    def train_one_epoch(self, loader, max_batches=None):
        self.model.train()
        losses, y_true, y_pred = [], [], []

        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="train", leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            losses.append(float(loss.item()))
            preds = torch.argmax(logits.detach(), dim=1)
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

        return {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else float("nan"),
        }

    @torch.no_grad()
    def evaluate(self, loader, max_batches=None):
        self.model.eval()
        losses, y_true, y_pred, y_prob = [], [], [], []

        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="eval", leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            losses.append(float(loss.item()))
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        return {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }

    def fit(self, train_loader, val_loader, epochs=3, max_batches_per_epoch=None, name="experiment"):
        history = []
        start = time.time()

        print(f"Training on device: {self.device}")

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch(train_loader, max_batches=max_batches_per_epoch)
            val_metrics = self.evaluate(val_loader)

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
            history.append(row)
            print(row)

            if self.scheduler is not None:
                self.scheduler.step()

        final_metrics = self.evaluate(val_loader)
        final_metrics["runtime_s"] = time.time() - start

        return ExperimentResult(
            name=name,
            history=history,
            metrics=final_metrics,
            y_true=final_metrics["y_true"],
            y_pred=final_metrics["y_pred"],
            y_prob=final_metrics["y_prob"],
        )


def run_transfer_experiment(
    name,
    model,
    train_loader,
    val_loader,
    epochs=3,
    learning_rate=1e-3,
    weight_decay=1e-4,
    device="auto",
    class_weights=None,
    label_smoothing=0.0,
    max_batches_per_epoch=None,
    scheduler=None,
):
    """Convenience wrapper used by the teaching notebooks."""
    trainer = Trainer(
        model,
        device=device,
        class_weights=class_weights,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        scheduler=scheduler,
        epochs_for_scheduler=epochs,
    )
    return trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        max_batches_per_epoch=max_batches_per_epoch,
        name=name,
    )

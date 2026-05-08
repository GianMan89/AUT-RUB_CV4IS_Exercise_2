
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


class Trainer:
    """Small training/evaluation engine for image classification."""
    def __init__(self, model, device="auto", class_weights=None, learning_rate=1e-3, weight_decay=1e-4):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        else:
            weight_tensor = None
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

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
        return {"loss": float(np.mean(losses)), "accuracy": float(accuracy_score(y_true, y_pred))}

    @torch.no_grad()
    def evaluate(self, loader, max_batches=None):
        self.model.eval()
        losses, y_true, y_pred = [], [], []
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="eval", leave=False)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            losses.append(float(loss.item()))
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        return {
            "loss": float(np.mean(losses)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "y_true": y_true,
            "y_pred": y_pred,
            "confusion_matrix": confusion_matrix(y_true, y_pred),
        }

    def fit(self, train_loader, val_loader, epochs=1, max_batches_per_epoch=None, name="experiment"):
        history = []
        start = time.time()
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_one_epoch(train_loader, max_batches=max_batches_per_epoch)
            val_metrics = self.evaluate(val_loader)
            row = {"epoch": epoch,
                   "train_loss": train_metrics["loss"], "train_acc": train_metrics["accuracy"],
                   "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"],
                   "val_macro_f1": val_metrics["macro_f1"]}
            history.append(row)
            print(row)
        final_metrics = self.evaluate(val_loader)
        final_metrics["runtime_s"] = time.time() - start
        return ExperimentResult(name=name, history=history, metrics=final_metrics,
                                y_true=final_metrics["y_true"], y_pred=final_metrics["y_pred"])

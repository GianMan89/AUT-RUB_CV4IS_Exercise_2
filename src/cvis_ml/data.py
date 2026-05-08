
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, List
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class DataBundle:
    """Container for train/validation loaders and dataset metadata."""
    train_loader: DataLoader
    val_loader: DataLoader
    class_names: List[str]
    num_classes: int


class CIFAR10DataModule:
    """Small guided image-classification dataset based on torchvision CIFAR-10.

    CIFAR-10 is not industrial, but it is convenient for first transfer-learning
    experiments because it is small, RGB, and directly loadable through torchvision.
    """
    def __init__(self, root="./data_cache", image_size=96, batch_size=32, num_workers=2,
                 max_train_samples=2000, max_val_samples=500, seed=42,
                 augment=True, weights_transforms=None):
        self.root = Path(root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.seed = seed
        self.augment = augment
        self.weights_transforms = weights_transforms

    def _make_transforms(self):
        # Default ImageNet-style preprocessing for pretrained CNNs.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_tf = []
        if self.augment:
            train_tf += [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(self.image_size, padding=8),
            ]
        else:
            train_tf += [transforms.Resize((self.image_size, self.image_size))]
        train_tf += [transforms.ToTensor(), normalize]
        val_tf = [transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor(), normalize]
        return transforms.Compose(train_tf), transforms.Compose(val_tf)

    @staticmethod
    def _balanced_subset_indices(dataset, max_samples, seed=42):
        if max_samples is None or max_samples >= len(dataset):
            return list(range(len(dataset)))
        rng = random.Random(seed)
        targets = np.array(dataset.targets)
        classes = sorted(set(targets.tolist()))
        per_class = max(1, max_samples // len(classes))
        indices = []
        for c in classes:
            c_idx = np.where(targets == c)[0].tolist()
            rng.shuffle(c_idx)
            indices.extend(c_idx[:per_class])
        rng.shuffle(indices)
        return indices[:max_samples]

    def setup(self):
        train_tf, val_tf = self._make_transforms()
        train_ds = datasets.CIFAR10(root=self.root, train=True, download=True, transform=train_tf)
        val_ds = datasets.CIFAR10(root=self.root, train=False, download=True, transform=val_tf)
        train_idx = self._balanced_subset_indices(train_ds, self.max_train_samples, self.seed)
        val_idx = self._balanced_subset_indices(val_ds, self.max_val_samples, self.seed + 1)
        train_subset = Subset(train_ds, train_idx)
        val_subset = Subset(val_ds, val_idx)
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True)
        return DataBundle(train_loader, val_loader, list(train_ds.classes), len(train_ds.classes))


class HFPILClassificationDataset(Dataset):
    """Torch Dataset wrapper for Hugging Face image classification datasets."""
    def __init__(self, hf_dataset, transform=None, label_transform=None):
        self.ds = hf_dataset
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[int(idx)]
        image = sample["image"].convert("RGB")
        label = int(sample["label"])
        if self.label_transform is not None:
            label = int(self.label_transform(label))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class MVTecCapsuleDataModule:
    """Industrial dataset module using a small Hugging Face MVTec Capsule subset.

    Important teaching note:
    The original MVTec AD setup is anomaly detection with defect-free training data.
    For this exercise we create a *supervised teaching split* from available normal
    and abnormal examples, so students can practice transfer learning for binary
    classification. This is not the official MVTec benchmark protocol.
    """
    def __init__(self, root="./data_cache", image_size=128, batch_size=16, num_workers=2,
                 max_train_samples=160, max_val_samples=80, seed=42, augment=True):
        self.root = Path(root)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.seed = seed
        self.augment = augment

    def _make_transforms(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_tf = [transforms.Resize((self.image_size, self.image_size))]
        if self.augment:
            train_tf += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
            ]
        train_tf += [transforms.ToTensor(), normalize]
        val_tf = [transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor(), normalize]
        return transforms.Compose(train_tf), transforms.Compose(val_tf)

    @staticmethod
    def _stratified_indices(labels, train_fraction=0.7, max_train=None, max_val=None, seed=42):
        rng = random.Random(seed)
        labels = np.asarray(labels)
        classes = sorted(set(labels.tolist()))
        train_idx, val_idx = [], []
        for c in classes:
            idx = np.where(labels == c)[0].tolist()
            rng.shuffle(idx)
            n_train = max(1, int(len(idx) * train_fraction))
            train_idx.extend(idx[:n_train])
            val_idx.extend(idx[n_train:])
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        if max_train is not None:
            train_idx = train_idx[:max_train]
        if max_val is not None:
            val_idx = val_idx[:max_val]
        return train_idx, val_idx

    def setup(self):
        try:
            from datasets import load_dataset, concatenate_datasets
        except Exception as e:
            raise ImportError("This data module requires the Hugging Face `datasets` package.") from e

        # This dataset uses a loading script. Pinning datasets==3.2.0 is recommended in requirements.
        train_hf = load_dataset("alexsu52/mvtec_capsule", split="train", trust_remote_code=True, cache_dir=str(self.root))
        test_hf = load_dataset("alexsu52/mvtec_capsule", split="test", trust_remote_code=True, cache_dir=str(self.root))
        all_hf = concatenate_datasets([train_hf, test_hf])
        labels = [int(x) for x in all_hf["label"]]
        train_idx, val_idx = self._stratified_indices(labels, max_train=self.max_train_samples,
                                                      max_val=self.max_val_samples, seed=self.seed)
        train_tf, val_tf = self._make_transforms()
        train_ds = HFPILClassificationDataset(all_hf.select(train_idx), transform=train_tf)
        val_ds = HFPILClassificationDataset(all_hf.select(val_idx), transform=val_tf)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True)
        class_names = ["normal", "abnormal"]
        return DataBundle(train_loader, val_loader, class_names, 2)

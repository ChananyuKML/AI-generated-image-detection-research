"""Shared model, strict image loading, metadata, and binary metrics."""

import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from networks.hog_cnn import DualCNN_128
from networks.learned_filters import LearnableFilterCNN128
from utils.custom_transforms import gray_center_224, raw_grad_224

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("learned_filter_cnn", "fixed_filter_cnn", "dual_cnn_128")
LABELS = {"nature": 0, "ai": 1}
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def dataset_path(data_root, dataset):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", dataset):
        raise ValueError("Dataset must be a folder name containing letters, digits, '-' or '_'.")
    return project_path(data_root) / dataset


def build_model(name):
    if name == "learned_filter_cnn":
        return LearnableFilterCNN128(trainable=True), gray_center_224
    if name == "fixed_filter_cnn":
        return LearnableFilterCNN128(trainable=False), gray_center_224
    if name == "dual_cnn_128":
        return DualCNN_128(), raw_grad_224
    raise ValueError(f"Unsupported model: {name}")


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install CUDA PyTorch and check the NVIDIA driver, or select --device cpu.")
    device = torch.device(name)
    torch.ones(1, device=device).sum().item()
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class BinaryImages(Dataset):
    """Stable file order; unreadable images fail instead of changing samples."""

    def __init__(self, root, transform):
        self.root, self.transform = Path(root), transform
        self.samples, self.counts = [], {}
        for label_name, label in LABELS.items():
            paths = sorted(path for path in (self.root / label_name).rglob("*")
                           if path.is_file() and path.suffix.lower() in EXTENSIONS)
            if not paths:
                raise ValueError(f"Missing or empty image folder: {self.root / label_name}")
            self.counts[label_name] = len(paths)
            self.samples.extend((path, label) for path in paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite image features: {path}")
        return tensor, label


def training_data_signature(root, datasets):
    """Hash prepared provenance; custom folders use names, labels, and sizes."""
    manifest = Path(root) / "manifest.csv"
    if manifest.is_file():
        return {"kind": "manifest_sha256", "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    digest = hashlib.sha256()
    for split, dataset in datasets.items():
        for path, label in dataset.samples:
            digest.update(f"{split}/{path.relative_to(dataset.root).as_posix()}:{label}:{path.stat().st_size}\n".encode())
    return {"kind": "paths_labels_sizes_sha256", "sha256": digest.hexdigest()}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(value, temporary)
    temporary.replace(path)


def binary_metrics(labels, scores, threshold=0.5):
    labels, scores = np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != scores.shape or not len(labels):
        raise ValueError("Metrics require equally sized, nonempty label and score vectors.")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and scores finite.")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Both classes are required for ROC-AUC and balanced accuracy.")
    predictions = scores > threshold
    tp, tn = int((predictions & (labels == 1)).sum()), int((~predictions & (labels == 0)).sum())
    fp, fn = negatives - tn, positives - tp
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    # Group equal scores once so ROC and AP use valid decision thresholds.
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(labels) - 1]
    true_positives = np.cumsum(sorted_labels)[ends]
    false_positives = ends + 1 - true_positives
    tpr, fpr = np.r_[0.0, true_positives / positives], np.r_[0.0, false_positives / negatives]
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    average_precision = float(np.sum(np.diff(tpr) * (true_positives / (ends + 1))))
    return {"images": len(labels), "threshold": threshold,
            "accuracy": (tp + tn) / len(labels),
            "balanced_accuracy": (tp / positives + tn / negatives) / 2,
            "roc_auc": auc, "average_precision": average_precision,
            "real_recall": tn / negatives, "ai_recall": tp / positives,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "positive_class": "ai", "label_map": LABELS}

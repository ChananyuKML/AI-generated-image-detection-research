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
from networks.kcnet import FixedFilterKCNet
from utils.custom_transforms import gray_center_224, raw_grad_224

ROOT = Path(__file__).resolve().parents[1]
KC_MODELS = ("fixed_filter_kcnet", "finetuned_cnn_kcnet")
MODELS = ("learned_filter_cnn", "fixed_filter_cnn", "dual_cnn_128", *KC_MODELS)
LABELS = {"nature": 0, "ai": 1}
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def dataset_path(data_root, dataset):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", dataset):
        raise ValueError("Dataset must be a folder name containing letters, digits, '-' or '_'.")
    return project_path(data_root) / dataset


def build_model(name, model_config=None):
    if name in KC_MODELS:
        return FixedFilterKCNet(train_cnn=name == "finetuned_cnn_kcnet", **(model_config or {})), gray_center_224
    if model_config:
        raise ValueError("Architecture options are only supported for KC models")
    if name == "learned_filter_cnn":
        return LearnableFilterCNN128(trainable=True), gray_center_224
    if name == "fixed_filter_cnn":
        return LearnableFilterCNN128(trainable=False), gray_center_224
    if name == "dual_cnn_128":
        return DualCNN_128(), raw_grad_224
    raise ValueError(f"Unsupported model: {name}")


def initialize_kc_encoder(model, checkpoint_path, dataset, signature):
    """Copy frontend/CNN from a validation-selected fixed-filter checkpoint."""
    if not isinstance(model, FixedFilterKCNet):
        raise TypeError("Encoder initialization requires a KC model")
    path = project_path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    if checkpoint.get("format_version") != 1 or config.get("model") != "fixed_filter_cnn":
        raise ValueError("Use a current fixed_filter_cnn checkpoint to initialize the KC encoder")
    if (config.get("selection") != "minimum validation BCE"
            or checkpoint.get("epoch") != checkpoint.get("best_epoch")
            or checkpoint.get("epoch", 0) < 1):
        raise ValueError("The encoder must be validation-selected best.pt, not an arbitrary last.pt")
    if config.get("dataset") != dataset or config.get("dataset_signature") != signature:
        raise ValueError("Encoder dataset/signature must match this controlled head experiment")
    if (config.get("input_size") != model.img_size or config.get("labels") != LABELS
            or config.get("transform") != "gray_center_224"):
        raise ValueError("Encoder input/label configuration does not match the KC model")
    state = checkpoint["model_state"]
    components = {}
    for prefix, module in (("frontend.", model.frontend), ("cnn_branch.", model.cnn_branch)):
        values = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
        expected = module.state_dict()
        if values.keys() != expected.keys() or any(values[key].shape != expected[key].shape for key in expected):
            raise ValueError(f"Encoder checkpoint has incompatible {prefix} state")
        if not all(torch.isfinite(value).all() for value in values.values()):
            raise ValueError("Encoder checkpoint contains non-finite values")
        components[prefix] = values
    for prefix, module in (("frontend.", model.frontend), ("cnn_branch.", model.cnn_branch)):
        module.load_state_dict(components[prefix], strict=True)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
            "sha256": digest, "model": config["model"], "epoch": checkpoint["epoch"],
            "selection": config["selection"], "training_dataset": config["dataset"],
            "dataset_signature": config["dataset_signature"], "seed": config["seed"]}


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

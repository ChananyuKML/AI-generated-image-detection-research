"""Fixed-threshold held-out metrics and per-image predictions."""

import argparse
import csv
import hashlib
import json

import torch
from torch.utils.data import DataLoader

from .experiment import (BinaryImages, binary_metrics, build_model, dataset_path,
                         project_path, select_device, write_json)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", help="Default: training dataset; override for cross-dataset testing")
    parser.add_argument("--data-root", default="dataset/prepared")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", help="Default: checkpoint-folder/evaluation/<dataset>/<split>")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Batch size must be positive and workers nonnegative.")
    device = select_device(args.device)
    checkpoint_path = project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    if checkpoint.get("format_version") != 1:
        raise ValueError("Use best.pt or last.pt from current train.py. Legacy raw .pth files use legacy/test_binary.py.")
    config = checkpoint["config"]
    name = args.dataset or config["dataset"]
    model, transform = build_model(config["model"], config.get("model_config"))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    root = dataset_path(args.data_root, name)
    dataset = BinaryImages(root / args.split, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    labels, scores, loss_sum = [], [], 0.0
    with torch.inference_mode():
        for inputs, targets in loader:
            logits = model(inputs.to(device, non_blocking=True)).flatten()
            loss_sum += torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets.to(device).float(), reduction="sum").item()
            labels.extend(targets.tolist())
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    metrics = binary_metrics(labels, scores, threshold=0.5)
    with checkpoint_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    metrics.update({"bce": loss_sum / len(dataset), "training_dataset": config["dataset"],
                    "evaluation_dataset": name, "split": args.split, "model": config["model"],
                    "seed": config["seed"], "checkpoint_epoch": checkpoint["epoch"],
                    "checkpoint": str(checkpoint_path), "checkpoint_sha256": digest,
                    "checkpoint_selection": config["selection"],
                    "threshold_selection": "fixed at 0.5; no test-set tuning"})
    if "encoder_source" in config:
        metrics.update(encoder_source=config["encoder_source"], model_config=config["model_config"])
    output_dir = project_path(args.output_dir) if args.output_dir else checkpoint_path.parent / "evaluation" / name / args.split
    write_json(output_dir / "metrics.json", metrics)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "label", "probability_ai", "prediction"))
        for (path, label), score in zip(dataset.samples, scores):
            writer.writerow((path.relative_to(root).as_posix(), label, score, int(score > 0.5)))
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Evaluation saved to {output_dir}", flush=True)

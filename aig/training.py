"""Training with separate seeded runs, validation selection, and epoch resume."""

import argparse
import csv
from datetime import datetime, timezone
import math
import platform
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .experiment import (ROOT, MODELS, BinaryImages, atomic_save, build_model,
                         dataset_path, project_path, seed_everything, seed_worker,
                         select_device, training_data_signature, write_json)


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; stopped without accepting this epoch.")
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            loss_sum += loss.item() * len(inputs)
            correct += int(((logits > 0) == labels.bool()).sum().item())
            total += len(inputs)
    return loss_sum / total, correct / total


def rng_state(generator):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), *state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    values = state["numpy"]
    np.random.set_state((values[0], np.array(values[1], dtype=np.uint32), *values[2:]))
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", "--network", choices=MODELS, default="learned_filter_cnn")
    result.add_argument("--dataset", default="progan_poc")
    result.add_argument("--data-root", "--dataroot", default="dataset/prepared")
    result.add_argument("--epochs", type=int, default=20, help="Total epochs, including those already completed when resuming")
    result.add_argument("--batch-size", "--batch_size", type=int, default=32)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--workers", "--num_workers", type=int, default=0)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    result.add_argument("--run-dir", help="Default: runs/<model>/<dataset>/seed_<seed>")
    result.add_argument("--resume", help="Resume an epoch checkpoint (normally <run-dir>/last.pt)")
    result.add_argument("--check-only", action="store_true")
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if (args.epochs < 1 or args.batch_size < 1 or args.workers < 0
            or not 0 <= args.seed < 2**32 or not math.isfinite(args.lr) or args.lr <= 0):
        argument_parser.error("Use positive epochs, batch size, and finite learning rate; workers >= 0; seed in [0, 2**32).")
    device = select_device(args.device)
    seed_everything(args.seed)
    model, transform = build_model(args.model)
    root = dataset_path(args.data_root, args.dataset)
    datasets = {split: BinaryImages(root / split, transform) for split in ("train", "val")}
    signature = training_data_signature(root, datasets)
    print(f"PyTorch {torch.__version__} | device={device} | model={args.model} | seed={args.seed}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    for split, dataset in datasets.items():
        print(f"{split}: {dataset.counts} | {len(dataset)} images", flush=True)
        inputs = dataset[0][0].unsqueeze(0).to(device)
        model.to(device).eval()
        with torch.no_grad():
            if not torch.isfinite(model(inputs)).all():
                raise FloatingPointError("Non-finite output during the pre-training check.")
    if args.check_only:
        print("Environment, dataset folders, image loading, and model forward checks passed.", flush=True)
        return
    run_dir = project_path(args.run_dir or f"runs/{args.model}/{args.dataset}/seed_{args.seed}")
    resume_path = project_path(args.resume) if args.resume else None
    if run_dir.exists() and any(run_dir.iterdir()) and resume_path is None:
        raise FileExistsError(f"Run directory already contains results: {run_dir}. Use --resume or another --run-dir.")
    if resume_path and resume_path.parent != run_dir:
        raise ValueError("Resume into the checkpoint's own run directory; use --run-dir to select it.")
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=split == "train", num_workers=args.workers,
                                pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                                generator=generator if split == "train" else None)
               for split, dataset in datasets.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    config = {"model": args.model, "dataset": args.dataset, "data_root": args.data_root,
              "seed": args.seed, "batch_size": args.batch_size, "lr": args.lr,
              "workers": args.workers, "epochs": args.epochs, "input_size": 224,
              "transform": "raw_grad_224" if args.model == "dual_cnn_128" else "gray_center_224",
              "labels": {"nature": 0, "ai": 1}, "dataset_signature": signature,
              "selection": "minimum validation BCE", "threshold": 0.5}
    start_epoch, best_loss, best_epoch, history = 0, float("inf"), 0, []
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        for key in ("model", "dataset", "seed", "batch_size", "lr", "dataset_signature"):
            if config[key] != checkpoint["config"][key]:
                raise ValueError(f"Resume configuration mismatch for {key}.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"]
        best_loss, best_epoch, history = checkpoint["best_val_loss"], checkpoint["best_epoch"], checkpoint["history"]
        restore_rng(checkpoint["rng"], generator)
        if args.epochs <= start_epoch:
            raise ValueError(f"Already completed {start_epoch} epochs; --epochs must be greater to resume.")
        best_file = run_dir / "best.pt"
        if not best_file.is_file() or torch.load(best_file, map_location="cpu", weights_only=True)["epoch"] != best_epoch:
            raise ValueError("Matching best.pt is required alongside a resumed checkpoint. Copy the complete run folder.")
        print(f"Resuming after epoch {start_epoch}", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    environment = {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                   "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
                   "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "platform": platform.platform(), "git_commit": git_commit}
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment)
    print(f"Results: {run_dir}", flush=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        started = time.monotonic()
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device)
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
               "val_loss": val_loss, "val_accuracy": val_acc, "seconds": time.monotonic() - started}
        history.append(row)
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch = val_loss, epoch
        checkpoint = {"format_version": 1, "model_state": model.state_dict(),
                      "optimizer_state": optimizer.state_dict(), "epoch": epoch,
                      "best_val_loss": best_loss, "best_epoch": best_epoch,
                      "config": config, "history": history, "rng": rng_state(generator)}
        if improved:
            atomic_save(checkpoint, run_dir / "best.pt")
        atomic_save(checkpoint, run_dir / "last.pt")
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch}/{args.epochs} | train BCE={train_loss:.5f}, acc={train_acc:.4f} | "
              f"val BCE={val_loss:.5f}, acc={val_acc:.4f} | {row['seconds']:.1f}s" +
              (" | saved best.pt" if improved else ""), flush=True)
    write_json(run_dir / "summary.json", {"completed_epochs": args.epochs,
               "best_epoch": best_epoch, "best_val_loss": best_loss, "config": config})
    print(f"Training complete. Best validation epoch: {best_epoch}. Evaluate {run_dir / 'best.pt'} on the test split.", flush=True)

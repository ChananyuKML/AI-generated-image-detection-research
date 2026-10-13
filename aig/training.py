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

from .experiment import (ROOT, MODELS, KC_MODELS, BinaryImages, atomic_save, build_model,
                         dataset_path, initialize_kc_encoder, project_path, seed_everything, seed_worker,
                         select_device, training_data_signature, write_json)
from networks.kcnet import KCNET_DEFAULTS
from .early_stopping import EarlyStopping
from .progress import TrainingProgress

KC_ARGUMENTS = {"pool_size": "kc_pool_size", "kc_units": "kc_units", "fan_in": "kc_fan_in",
                "projection_seed": "projection_seed", "feature_eps": "kc_feature_eps"}


def resolve_early_stopping_config(args, saved_config=None):
    settings = {"patience": 0, "min_delta": 0.0}
    if saved_config is not None:
        if set(saved_config) != set(settings):
            raise ValueError("Checkpoint has unsupported early-stopping settings")
        settings.update(saved_config)
    for name in settings:
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    EarlyStopping(**settings)
    return settings


def resolve_model_config(args, saved_config=None):
    supplied = {key: getattr(args, option) for key, option in KC_ARGUMENTS.items()
                if getattr(args, option) is not None}
    if args.model not in KC_MODELS:
        if supplied or args.encoder_checkpoint:
            raise ValueError("KC options and --encoder-checkpoint require a KC model")
        return None
    if saved_config is not None:
        if set(saved_config) != set(KCNET_DEFAULTS):
            raise ValueError("KC checkpoint has missing or unsupported architecture options")
        for key, value in supplied.items():
            if value != saved_config[key]:
                raise ValueError(f"Resume configuration mismatch for {key}.")
        return dict(saved_config)
    return dict(KCNET_DEFAULTS, **supplied)


def run_epoch(model, loader, criterion, device, optimizer=None, progress=None):
    model.train(optimizer is not None)
    loss_sum, correct, total = 0.0, 0, 0
    if progress is not None:
        progress.start_phase("training" if optimizer is not None else "validation", len(loader))
    with torch.set_grad_enabled(optimizer is not None):
        for batch, (inputs, labels) in enumerate(loader, 1):
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
            if progress is not None:
                progress.batch("training" if optimizer is not None else "validation",
                               batch, len(loader), total, loss_sum / total, correct / total)
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
    result.add_argument("--patience", type=int, help="Stop after N epochs without significant validation BCE improvement; 0 disables (fresh default), restored on resume")
    result.add_argument("--min-delta", type=float, help="Minimum validation BCE decrease that resets patience (default 0; restored on resume)")
    kc = result.add_argument_group("KC model options")
    kc.add_argument("--encoder-checkpoint", help="Validation-selected fixed_filter_cnn best.pt; required for fresh KC runs")
    kc.add_argument("--kc-units", type=int, help="Kenyon units (default 4096; restored on resume)")
    kc.add_argument("--kc-fan-in", type=int, help="Distinct input features per unit (default 7)")
    kc.add_argument("--projection-seed", type=int, help="Seed for fixed random connections (default 42)")
    kc.add_argument("--kc-pool-size", type=int, help="Spatial pooling size (default 4)")
    kc.add_argument("--kc-feature-eps", type=float, help="Safe L2 normalization floor (default 1e-8)")
    return result


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if (args.epochs < 1 or args.batch_size < 1 or args.workers < 0
            or not 0 <= args.seed < 2**32 or not math.isfinite(args.lr) or args.lr <= 0):
        argument_parser.error("Use positive epochs, batch size, and finite learning rate; workers >= 0; seed in [0, 2**32).")
    device = select_device(args.device)
    resume_path = project_path(args.resume) if args.resume else None
    resumed_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True) if resume_path else None
    if resumed_checkpoint is not None and resumed_checkpoint.get("format_version") != 1:
        raise ValueError("Resume requires a structured checkpoint from current train.py")
    early_settings = resolve_early_stopping_config(
        args, resumed_checkpoint["config"].get("early_stopping") if resumed_checkpoint else None)
    if args.model in KC_MODELS and resumed_checkpoint is not None:
        if resumed_checkpoint["config"]["model"] != args.model:
            raise ValueError("Resume configuration mismatch for model.")
        if "model_config" not in resumed_checkpoint["config"]:
            raise ValueError("KC checkpoint is missing its architecture configuration")
    saved_model_config = resumed_checkpoint["config"].get("model_config") if resumed_checkpoint else None
    model_config = resolve_model_config(args, saved_model_config)
    seed_everything(args.seed)
    model, transform = build_model(args.model, model_config)
    root = dataset_path(args.data_root, args.dataset)
    datasets = {split: BinaryImages(root / split, transform) for split in ("train", "val")}
    signature = training_data_signature(root, datasets)
    encoder_source = None
    if args.model in KC_MODELS:
        if resumed_checkpoint is not None:
            if args.encoder_checkpoint:
                raise ValueError("Resume uses the saved encoder; omit --encoder-checkpoint")
            encoder_source = resumed_checkpoint["config"]["encoder_source"]
            model.load_state_dict(resumed_checkpoint["model_state"])
        else:
            if not args.encoder_checkpoint:
                raise ValueError("Fresh KC runs require --encoder-checkpoint")
            encoder_source = initialize_kc_encoder(model, args.encoder_checkpoint, args.dataset, signature)
    print(f"PyTorch {torch.__version__} | device={device} | model={args.model} | seed={args.seed}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    if encoder_source:
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        cnn_mode = "trainable" if model.train_cnn else "frozen"
        print(f"Encoder source: epoch {encoder_source['epoch']} | CNN={cnn_mode} | SHA-256={encoder_source['sha256']} | "
              f"trainable parameters={trainable} | projection seed={model_config['projection_seed']}", flush=True)
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
    parameters = (parameter for parameter in model.parameters() if parameter.requires_grad) if encoder_source else model.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    config = {"model": args.model, "dataset": args.dataset, "data_root": args.data_root,
              "seed": args.seed, "batch_size": args.batch_size, "lr": args.lr,
              "workers": args.workers, "epochs": args.epochs, "input_size": 224,
              "transform": "raw_grad_224" if args.model == "dual_cnn_128" else "gray_center_224",
              "labels": {"nature": 0, "ai": 1}, "dataset_signature": signature,
              "selection": "minimum validation BCE", "threshold": 0.5,
              "early_stopping": early_settings}
    if encoder_source:
        config.update(model_config=model_config, encoder_source=encoder_source,
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    start_epoch, best_loss, best_epoch, history = 0, float("inf"), 0, []
    if resume_path:
        checkpoint = resumed_checkpoint
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
    saved_early_state = None
    if resumed_checkpoint and resumed_checkpoint["config"].get("early_stopping") == early_settings:
        saved_early_state = resumed_checkpoint.get("early_stopping_state")
    monitor = EarlyStopping.from_history(early_settings, history, saved_early_state)
    if monitor.should_stop:
        raise ValueError("Checkpoint already exhausted early-stopping patience. Explicitly increase --patience or use --patience 0 to continue.")
    if monitor.patience:
        print(f"Early stopping: patience={monitor.patience}, min_delta={monitor.min_delta:g}, "
              f"carried bad epochs={monitor.bad_epochs}", flush=True)
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
    completed_epochs, stopped_early = start_epoch, False
    progress = TrainingProgress(run_dir, args.model, args.dataset, args.seed, args.epochs, start_epoch)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        started = time.monotonic()
        progress.start_epoch(epoch)
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer, progress)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device, progress=progress)
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
               "val_loss": val_loss, "val_accuracy": val_acc, "seconds": time.monotonic() - started}
        history.append(row)
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch = val_loss, epoch
        completed_epochs = epoch
        stop = monitor.update(val_loss)
        checkpoint = {"format_version": 1, "model_state": model.state_dict(),
                      "optimizer_state": optimizer.state_dict(), "epoch": epoch,
                      "best_val_loss": best_loss, "best_epoch": best_epoch,
                      "config": config, "history": history, "rng": rng_state(generator),
                      "early_stopping_state": monitor.state_dict()}
        progress.emit(force=True, phase="saving_checkpoint")
        if improved:
            atomic_save(checkpoint, run_dir / "best.pt")
        atomic_save(checkpoint, run_dir / "last.pt")
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        progress.finish_epoch(row, best_epoch, best_loss, monitor.bad_epochs, monitor.patience)
        print(f"Epoch {epoch}/{args.epochs} | train BCE={train_loss:.5f}, acc={train_acc:.4f} | "
              f"val BCE={val_loss:.5f}, acc={val_acc:.4f} | {row['seconds']:.1f}s" +
              (" | saved best.pt" if improved else "") +
              (f" | patience {monitor.bad_epochs}/{monitor.patience}" if monitor.patience else ""), flush=True)
        if stop:
            stopped_early = epoch < args.epochs
            if stopped_early:
                print(f"Early stopping after epoch {epoch}: {monitor.bad_epochs} epochs without significant validation BCE improvement.", flush=True)
            break
    write_json(run_dir / "summary.json", {"completed_epochs": completed_epochs,
               "requested_epochs": args.epochs, "stopped_early": stopped_early,
               "stop_reason": "early_stopping" if stopped_early else "epoch_target",
               "early_stopping_state": monitor.state_dict(),
               "best_epoch": best_epoch, "best_val_loss": best_loss, "config": config})
    progress.finish(completed_epochs, stopped_early)
    print(f"Training complete after {completed_epochs}/{args.epochs} epochs. Best validation epoch: {best_epoch}. "
          f"Evaluate {run_dir / 'best.pt'} on the test split.", flush=True)

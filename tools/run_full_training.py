"""Durable full-dataset training queue with a read-only live progress dashboard."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("progan", "tiny_genimage", "cifake")
MODELS = ("fixed_filter_cnn", "learned_filter_cnn", "dual_cnn_128",
          "fixed_filter_kcnet", "finetuned_cnn_kcnet")
KC_MODELS = MODELS[-2:]


def pid_alive(pid):
    if not isinstance(pid, int) or pid < 1:
        return False
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel.OpenProcess(0x00101000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_jobs(datasets=DATASETS, models=MODELS, seeds=(42,)):
    if not set(datasets).issubset(DATASETS) or not set(models).issubset(MODELS):
        raise ValueError("This queue accepts only full datasets and supported binary models.")
    if len(set(datasets)) != len(datasets) or len(set(models)) != len(models) or len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate datasets, models, or seeds would reuse a run directory.")
    if any(model in KC_MODELS for model in models) and "fixed_filter_cnn" not in models:
        raise ValueError("Include fixed_filter_cnn so each KC job has its matched encoder.")
    return [{"id": f"{dataset}/{model}/seed_{seed}", "dataset": dataset, "model": model,
             "seed": seed, "status": "queued"}
            for dataset in datasets for seed in seeds for model in MODELS if model in models]


def inspect_dataset(name, data_root):
    root = Path(data_root) / name
    manifest = read_json(root / "manifest.json")
    if not manifest or manifest.get("variant") != name:
        raise ValueError(f"Missing matching prepared manifest for {name}: {root}")
    if (manifest.get("label_map") != {"nature": 0, "ai": 1}
            or manifest.get("exact_decoded_pixel_overlap_across_splits") != 0):
        raise ValueError(f"Unexpected binary-label or split-integrity metadata: {name}")
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {row["relative_path"] for row in rows}
    actual = {path.relative_to(root).as_posix() for split in ("train", "val", "test")
              for path in (root / split).rglob("*")
              if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png")}
    if actual != expected or len(expected) != len(rows):
        raise ValueError(f"Image inventory differs from the manifest: {name}")
    counts = Counter("/".join(row["relative_path"].split("/")[:2]) for row in rows)
    if dict(counts) != manifest["counts"] or any(not counts[f"{split}/{label}"]
                                               for split in ("train", "val", "test") for label in ("ai", "nature")):
        raise ValueError(f"Split/class counts differ from the manifest: {name}")
    return {"name": name, "counts": dict(counts), "images": len(rows),
            "signature": {"kind": "manifest_sha256", "sha256": hashlib.sha256((root / "manifest.csv").read_bytes()).hexdigest()}}


def snapshot_sources(batch):
    source = batch / "source"
    if source.is_dir():
        record = read_json(batch / "source_files.json")
        if not record:
            raise ValueError("Existing source snapshot has no hash record.")
        for name, digest in record.items():
            if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source snapshot changed: {name}")
        return source
    files = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT).decode().split("\0")
    hashes = {}
    for name in sorted(set(files) - {""}):
        path = ROOT / name
        if path.is_file() and (path.suffix.lower() in (".py", ".md", ".json", ".txt", ".ps1", ".yml", ".html") or name in (".gitignore", ".gitattributes")):
            destination = source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            hashes[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    if not (source / "train.py").is_file():
        raise ValueError("The source snapshot is missing train.py.")
    write_json(batch / "source_files.json", hashes)
    with (batch / "source_changes.patch").open("wb") as handle:
        subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=ROOT, stdout=handle, check=True)
    with (batch / "installed_requirements.txt").open("w", encoding="utf-8") as handle:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], cwd=ROOT, stdout=handle, check=True)
    return source


class TrainingQueue:
    def __init__(self, args):
        self.args = args
        self.batch = Path(args.batch_dir).expanduser()
        if not self.batch.is_absolute():
            self.batch = ROOT / self.batch
        self.batch = self.batch.resolve()
        if not self.batch.is_relative_to(ROOT / "runs"):
            raise ValueError("Batch files must be under the project's ignored runs folder.")
        self.data_root = (ROOT / "dataset" / "prepared").resolve()
        jobs = make_jobs(args.datasets, args.models, args.seeds)
        self.batch.mkdir(parents=True, exist_ok=True)
        self.lockfile = self.batch / "manager.lock"
        if self.lockfile.exists():
            owner = read_json(self.lockfile)
            if not owner or pid_alive(owner.get("pid")):
                raise RuntimeError("This batch already has a live manager or an unverifiable lock.")
            self.lockfile.unlink()
        descriptor = os.open(self.lockfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started_utc": utc_now()}, handle)
        self.plan = {"datasets": args.datasets, "models": [model for model in MODELS if model in args.models],
                     "seeds": args.seeds, "epochs": args.epochs, "patience": args.patience,
                     "min_delta": 0.0, "batch_size": args.batch_size, "lr": 1e-4, "workers": args.workers,
                     "device": "cuda", "data_root": str(self.data_root), "jobs": [job["id"] for job in jobs]}
        previous = read_json(self.batch / "plan.json")
        if previous and previous != self.plan:
            raise ValueError("Existing batch plan differs. Select a new --batch-dir.")
        write_json(self.batch / "plan.json", self.plan)
        self.state = {"status": "preparing", "jobs": jobs, "active_job": None,
                      "manager_pid": os.getpid(), "started_utc": utc_now(),
                      "dashboard_url": f"http://127.0.0.1:{args.port}/", "settings": self.plan,
                      "batch_dir": str(self.batch)}
        self.lock = threading.RLock()
        self.child = None
        self.source = None
        self.gpu_cache = None
        self.gpu_updated = 0
        self.save()

    def save(self):
        with self.lock:
            self.state["updated_utc"] = utc_now()
            write_json(self.batch / "status.json", self.state)

    def run_dir(self, job):
        return self.batch / "experiments" / job["model"] / job["dataset"] / f"seed_{job['seed']}"

    def expected_config(self, job):
        return {"model": job["model"], "dataset": job["dataset"], "seed": job["seed"],
                "batch_size": self.args.batch_size, "lr": 1e-4, "workers": self.args.workers,
                "input_size": 224,
                "dataset_signature": self.state["datasets"][job["dataset"]]["signature"],
                "early_stopping": {"patience": self.args.patience, "min_delta": 0.0}}

    def validate_config(self, job, config):
        for key, expected in self.expected_config(job).items():
            if config.get(key) != expected:
                raise ValueError(f"Existing {job['id']} configuration differs for {key}; results were preserved.")

    def train_arguments(self, job):
        run = self.run_dir(job)
        config = read_json(run / "config.json")
        if config:
            self.validate_config(job, config)
        summary = read_json(run / "summary.json")
        if summary:
            self.validate_config(job, summary["config"])
            if (summary["completed_epochs"] >= self.args.epochs or summary.get("stopped_early")):
                if not (run / "best.pt").is_file() or not (run / "last.pt").is_file():
                    raise ValueError("Completed run is missing checkpoints.")
                return None
        arguments = [str(self.source / "train.py"), "--model", job["model"], "--dataset", job["dataset"],
                     "--data-root", str(self.data_root), "--run-dir", str(run), "--epochs", str(self.args.epochs),
                     "--batch-size", str(self.args.batch_size), "--workers", str(self.args.workers),
                     "--seed", str(job["seed"]), "--device", "cuda", "--lr", "0.0001",
                     "--patience", str(self.args.patience), "--min-delta", "0"]
        if (run / "last.pt").is_file():
            import torch
            checkpoint = torch.load(run / "last.pt", weights_only=True, map_location="cpu")
            self.validate_config(job, checkpoint["config"])
            history = checkpoint["history"]
            bad_epochs, best_loss = 0, float("inf")
            for row in history:
                if row["val_loss"] < best_loss:
                    best_loss, bad_epochs = row["val_loss"], 0
                else:
                    bad_epochs += 1
            if checkpoint["epoch"] >= self.args.epochs or (self.args.patience and bad_epochs >= self.args.patience):
                if not (run / "best.pt").is_file():
                    raise ValueError("Interrupted stopping boundary is missing best.pt.")
                write_json(run / "summary.json", {"completed_epochs": checkpoint["epoch"],
                    "requested_epochs": self.args.epochs, "stopped_early": checkpoint["epoch"] < self.args.epochs,
                    "stop_reason": "early_stopping" if checkpoint["epoch"] < self.args.epochs else "epoch_target",
                    "best_epoch": checkpoint["best_epoch"], "best_val_loss": checkpoint["best_val_loss"],
                    "early_stopping_state": checkpoint.get("early_stopping_state"), "config": checkpoint["config"]})
                return None
            arguments += ["--resume", str(run / "last.pt")]
        elif run.exists() and any(run.iterdir()):
            raise ValueError(f"Partial run has no resumable checkpoint: {run}. Select a new batch folder.")
        elif job["model"] in KC_MODELS:
            encoder = dict(job, model="fixed_filter_cnn")
            checkpoint = self.run_dir(encoder) / "best.pt"
            if not checkpoint.is_file():
                raise ValueError(f"Matching fixed-filter encoder is missing: {checkpoint}")
            arguments += ["--encoder-checkpoint", str(checkpoint), "--projection-seed", "42"]
        return arguments

    def launch(self, job, arguments, phase):
        log = self.batch / "logs" / (job["id"].replace("/", "_") + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        job.update(status=phase, log=str(log.relative_to(ROOT)), started_utc=utc_now())
        environment = os.environ.copy()
        environment.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                           MPLBACKEND="Agg", MPLCONFIGDIR=str(self.batch / "matplotlib"))
        command = [sys.executable, "-u", *arguments]
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{utc_now()}] {subprocess.list2cmdline(command)}\n")
            handle.flush()
            with self.lock:
                self.child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                              stdout=handle, stderr=subprocess.STDOUT,
                                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                job["pid"] = self.child.pid
            self.save()
            while self.child.poll() is None:
                self.save()
                time.sleep(2)
            code = self.child.returncode
        job["pid"] = None
        self.save()
        if code:
            raise RuntimeError(f"{phase} failed with exit code {code}. See {log}")

    def run(self):
        prevented_sleep = False
        try:
            if os.name == "nt":
                import ctypes
                prevented_sleep = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for this full-training queue.")
            torch.ones(1, device="cuda").sum().item()
            self.state["environment"] = {"python": sys.version.split()[0], "torch": str(torch.__version__),
                                         "gpu": torch.cuda.get_device_name(0), "sleep_inhibited": prevented_sleep}
            self.state["datasets"] = {name: inspect_dataset(name, self.data_root) for name in self.args.datasets}
            self.source = snapshot_sources(self.batch)
            self.state["source_dir"] = str(self.source)
            self.state["status"] = "running"
            self.save()
            for job in self.state["jobs"]:
                with self.lock:
                    self.state["active_job"] = job["id"]
                self.save()
                arguments = self.train_arguments(job)
                if arguments:
                    self.launch(job, arguments, "training")
                job["summary"] = read_json(self.run_dir(job) / "summary.json")
                self.launch(job, [str(self.source / "evaluate.py"), "--checkpoint", str(self.run_dir(job) / "best.pt"),
                                  "--data-root", str(self.data_root), "--dataset", job["dataset"], "--device", "cuda",
                                  "--batch-size", str(self.args.batch_size), "--workers", str(self.args.workers)], "evaluating")
                metrics = read_json(self.run_dir(job) / "evaluation" / job["dataset"] / "test" / "metrics.json")
                if not metrics:
                    raise RuntimeError("Evaluation did not produce metrics.json.")
                job.update(status="completed", finished_utc=utc_now(), metrics=metrics)
                self.save()
                self.results()
            self.state.update(status="completed", active_job=None, finished_utc=utc_now())
            self.save()
        except BaseException as error:
            with self.lock:
                self.state.update(status="failed", error=str(error), traceback=traceback.format_exc())
                active = next((job for job in self.state["jobs"] if job["id"] == self.state["active_job"]), None)
                if active:
                    active.update(status="failed", error=str(error))
            self.save()
            print(traceback.format_exc(), flush=True)
        finally:
            if prevented_sleep:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

    def results(self):
        rows = []
        for job in self.state["jobs"]:
            if job["status"] == "completed":
                metrics, summary = job["metrics"], job["summary"]
                rows.append({"dataset": job["dataset"], "model": job["model"], "seed": job["seed"],
                             "completed_epochs": summary["completed_epochs"], "best_epoch": summary["best_epoch"],
                             **{key: metrics[key] for key in ("accuracy", "balanced_accuracy", "roc_auc", "average_precision", "bce")}})
        if rows:
            with (self.batch / "results.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def gpu(self):
        if time.monotonic() - self.gpu_updated > 8:
            self.gpu_updated = time.monotonic()
            try:
                output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, timeout=5)
                utilization, used, total = output.splitlines()[0].split(",")
                self.gpu_cache = {"utilization": int(utilization), "memory_used_mib": int(used), "memory_total_mib": int(total)}
            except (OSError, ValueError, subprocess.SubprocessError):
                self.gpu_cache = None
        return self.gpu_cache

    def view(self):
        with self.lock:
            value = json.loads(json.dumps(self.state))
            value["child_alive"] = self.child is not None and self.child.poll() is None
        value["server_utc"] = utc_now()
        value["gpu"] = self.gpu()
        active = next((job for job in value["jobs"] if job["id"] == value["active_job"]), None)
        if active:
            run = self.run_dir(active)
            value["progress"] = read_json(run / "progress.json")
            if (active.get("status") == "training" and value["progress"]
                    and value["progress"].get("pid") != active.get("pid")):
                value["progress"] = None
            try:
                with (run / "history.csv").open(newline="", encoding="utf-8") as handle:
                    value["history"] = list(csv.DictReader(handle))
            except OSError:
                value["history"] = []
            if active.get("log"):
                try:
                    with (ROOT / active["log"]).open("rb") as handle:
                        handle.seek(max(0, handle.seek(0, 2) - 12000))
                        value["log_tail"] = handle.read().decode("utf-8", errors="replace")
                except OSError:
                    value["log_tail"] = ""
        return value


def make_handler(queue):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                payload = (ROOT / "tools" / "training_dashboard.html").read_bytes()
                content_type = "text/html; charset=utf-8"
            elif self.path == "/api/status":
                payload = json.dumps(queue.view(), allow_nan=False).encode()
                content_type = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", default="runs/full_comparison")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if (args.epochs < 1 or args.patience < 0 or args.batch_size < 1 or args.workers < 0
            or not 1024 <= args.port <= 65535 or any(not 0 <= seed < 2**32 for seed in args.seeds)):
        parser.error("Invalid epoch, patience, batch, workers, port, or seed setting.")
    # Bind before creating a queue: an occupied port prevents a duplicate manager.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), BaseHTTPRequestHandler)
    queue = TrainingQueue(args)
    server.RequestHandlerClass = make_handler(queue)
    worker = threading.Thread(target=queue.run, daemon=True)
    worker.start()
    print(f"Live training: http://127.0.0.1:{args.port}/ | batch={queue.batch}", flush=True)
    try:
        server.serve_forever(poll_interval=.5)
    finally:
        if queue.child is not None and queue.child.poll() is None:
            queue.child.terminate()
            queue.child.wait(timeout=30)
        server.server_close()
        owner = read_json(queue.lockfile)
        if owner and owner.get("pid") == os.getpid():
            queue.lockfile.unlink()


if __name__ == "__main__":
    main()

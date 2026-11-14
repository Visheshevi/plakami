"""Train YOLOX-Tiny from a YAML config into `<runs_dir>/<run_id>/`.

Runs on a CUDA GPU with YOLOX's own trainer (Colab), or on an Apple-silicon GPU (MPS) with
`local_trainer.py` (ADR-0012). `--device cpu` exists for tests only.

Records the git state, config, dataset/label hashes, YOLOX commit and starting checkpoint
hash in `run_info.json` before training, and the final checkpoint hash afterwards. If the
run folder already has `yolox/latest_ckpt.pth`, training resumes from it (Colab sessions
can end at any time).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import platform
import random
import shutil
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from platedet.eval.baseline import _git_state, _prepare_model, sha256_file
from platedet.train.dataset import training_dir
from platedet.train.exp import make_exp
from platedet.train.local_trainer import make_local_trainer, mps_type_names, pick_device
from platedet.train.yolox_src import YOLOX_COMMIT

LOGGER = logging.getLogger("platedet.train")
EXPERIMENT_NAME = "yolox"


def _run_git_state() -> dict[str, Any]:
    """Git state, or the commit stamped into a Colab bundle when there is no .git folder."""
    state = _git_state()
    bundle = Path("bundle_info.json")
    if state["commit"] == "unknown" and bundle.is_file():
        stamp = json.loads(bundle.read_text(encoding="utf-8"))
        return {"commit": stamp["git_commit"], "dirty": stamp["dirty"], "source": "colab bundle"}
    return state


def run_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """Hashes of everything the run depends on (dataset, labels, start checkpoint)."""
    data_dir = training_dir(config)
    stats = json.loads((data_dir / "stats.json").read_text(encoding="utf-8"))
    if int(stats["input_size"]) != int(config["input_size"]):
        raise ValueError(
            f"{data_dir} was built for {stats['input_size']}px, config wants {config['input_size']}"
        )
    return {
        "combined_manifest_sha256": stats["combined_manifest_sha256"],
        "pseudo_labels_sha256": stats["pseudo_labels_sha256"],
        "train_annotations_sha256": sha256_file(data_dir / "annotations" / "train.json"),
        "val_annotations_sha256": sha256_file(data_dir / "annotations" / "val.json"),
        "training_data_stats": stats,
        "yolox_commit": YOLOX_COMMIT,
        "pretrained_url": config["pretrained"]["url"],
        "pretrained_sha256": config["pretrained"]["sha256"],
    }


def _environment(torch: Any, device: str) -> dict[str, Any]:
    environment: dict[str, Any] = {"torch": torch.__version__, "device": device}
    if device == "cuda":
        environment |= {"cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    elif device == "mps":
        environment["gpu"] = platform.processor() or "apple-silicon"
        environment["machine"] = platform.platform()
    return environment


def train(
    config_path: Path,
    *,
    runs_dir: Path,
    run_id: str | None = None,
    device: str | None = None,
) -> Path:
    import torch

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_id = run_id or f"{config['name']}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    run_dir = runs_dir / run_id
    latest = run_dir / EXPERIMENT_NAME / "latest_ckpt.pth"
    resume = latest.is_file()
    if device is None:
        device = pick_device()
        if device == "cpu":
            raise RuntimeError("No CUDA or Apple GPU found (pass --device cpu for tests only)")

    pretrained = _prepare_model(config["pretrained"])
    inputs = run_inputs(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    info_path = run_dir / "run_info.json"
    if resume:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info["inputs"] != inputs:
            raise ValueError("Resuming with different data or checkpoint than the run started with")
        info.setdefault("resumed_at", []).append(
            {"at": datetime.now(UTC).isoformat(), "environment": _environment(torch, device)}
        )
    else:
        shutil.copy(config_path, run_dir / "config.yaml")
        info = {
            "run_id": run_id,
            "started_at": datetime.now(UTC).isoformat(),
            "git": _run_git_state(),
            "config": config,
            "inputs": inputs,
            "environment": _environment(torch, device),
        }
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    seed = int(config.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    exp = make_exp(config, run_dir)
    from yolox.core import Trainer
    from yolox.utils import configure_nccl, configure_omp

    training = config.get("train", {})
    args = Namespace(
        experiment_name=EXPERIMENT_NAME,
        name=None,
        dist_backend="nccl",
        dist_url=None,
        batch_size=int(training.get("batch_size", 32)),
        devices=1,
        exp_file=None,
        resume=resume,
        ckpt=None if resume else str(pretrained),
        start_epoch=None,
        num_machines=1,
        machine_rank=0,
        fp16=bool(training.get("fp16", True)),
        cache=None,
        occupy=False,
        logger="tensorboard",
        opts=[],
    )
    LOGGER.info("%s run %s on %s", "Resuming" if resume else "Starting", run_dir, device)
    if device == "cuda":
        configure_nccl()
        configure_omp()
        torch.backends.cudnn.benchmark = True
        Trainer(exp, args).train()
    else:
        with mps_type_names() if device == "mps" else contextlib.nullcontext():
            make_local_trainer(exp, args, device).train()

    best = run_dir / EXPERIMENT_NAME / "best_ckpt.pth"
    info["finished_at"] = datetime.now(UTC).isoformat()
    info["best_checkpoint"] = str(best.relative_to(run_dir))
    info["best_checkpoint_sha256"] = sha256_file(best)
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-id", default=None, help="fixed ID so a later call resumes")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(train(args.config, runs_dir=args.runs_dir, run_id=args.run_id, device=args.device))


if __name__ == "__main__":
    main()

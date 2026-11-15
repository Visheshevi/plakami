"""Evaluate an off-the-shelf ONNX YOLOX model on our splits (T2 baseline).

Read-only evaluation: fixed thresholds from SPEC §6, nothing tuned, so running on
`test` is allowed. Each run writes `runs/<run_id>/` with the config, summary.json
(commit, manifest hash, checkpoint hash, metrics, worst misses), raw predictions and a
gallery of the largest missed boxes per split and class.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

from platedet.data.open_images import download_url
from platedet.eval.metrics import (
    CLASS_NAMES,
    GroundTruth,
    ImageResult,
    missed_boxes,
    summarize,
)
from platedet.postprocess import (
    DEFAULT_NMS_IOU,
    Detection,
    detect_tiled,
    letterbox,
    postprocess,
)

LOGGER = logging.getLogger("platedet.eval")
EVAL_SPLITS = ("train", "val", "test", "holdout")
Predictor = Callable[[Image.Image], list[Detection]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(dataset_dir: Path, split: str) -> list[ImageResult]:
    """Read one COCO split into per-image ground truth (boxes as xyxy pixels)."""
    dataset = json.loads((dataset_dir / "annotations" / f"{split}.json").read_text("utf-8"))
    names_to_ids = {category["name"]: category["id"] for category in dataset["categories"]}
    images: dict[int, ImageResult] = {}
    for image in dataset["images"]:
        annotated = image.get("annotated_classes", list(names_to_ids))
        images[image["id"]] = ImageResult(
            file_name=image["file_name"],
            annotated_classes=frozenset(names_to_ids[name] for name in annotated),
            ground_truth=[],
        )
    for annotation in dataset["annotations"]:
        x, y, width, height = annotation["bbox"]
        images[annotation["image_id"]].ground_truth.append(
            GroundTruth(
                x,
                y,
                x + width,
                y + height,
                int(annotation["category_id"]),
                int(annotation["id"]) if "id" in annotation else None,
            )
        )
    return list(images.values())


class OnnxYoloxPredictor:
    """Runs a YOLOX ONNX model (raw head outputs, NMS outside the graph)."""

    def __init__(
        self,
        model_path: Path,
        *,
        input_size: int,
        channel_order: str,
        conf_threshold: float,
        nms_iou: float,
        class_map: dict[int, int] | None,
    ) -> None:
        import onnxruntime

        self.session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.channel_order = channel_order
        self.conf_threshold = conf_threshold
        self.nms_iou = nms_iou
        self.class_map = class_map

    def __call__(self, image: Image.Image) -> list[Detection]:
        tensor, ratio = letterbox(image, self.input_size, self.channel_order)
        raw = self.session.run(None, {self.input_name: tensor})[0]
        return postprocess(
            raw,
            size=self.input_size,
            ratio=ratio,
            image_width=image.width,
            image_height=image.height,
            conf_threshold=self.conf_threshold,
            nms_iou=self.nms_iou,
            class_map=self.class_map,
        )


class TiledPredictor:
    """Whole image plus `grid` x `grid` tiles through another predictor (ADR-0013)."""

    def __init__(self, predictor: Predictor, *, grid: int, overlap: float, nms_iou: float) -> None:
        self.predictor = predictor
        self.grid = grid
        self.overlap = overlap
        self.nms_iou = nms_iou

    def __call__(self, image: Image.Image) -> list[Detection]:
        return detect_tiled(
            image, self.predictor, grid=self.grid, overlap=self.overlap, nms_iou=self.nms_iou
        )


def _prepare_model(model: dict[str, Any]) -> Path:
    """Download the checkpoint if missing and refuse it unless the SHA-256 matches."""
    path = Path(model["path"])
    if not path.is_file():
        download_url(str(model["url"]), path)
    actual = sha256_file(path)
    if actual != model["sha256"]:
        raise ValueError(f"Checkpoint SHA-256 mismatch for {path}: {actual}")
    return path


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}
    return {"commit": commit, "dirty": dirty}


def write_gallery(
    misses: list[tuple[ImageResult, GroundTruth]], dataset_dir: Path, path: Path, limit: int
) -> list[dict[str, Any]]:
    """Save a numbered contact sheet of missed boxes (with context); return their details."""
    columns, cell_width, cell_height = 6, 240, 150
    chosen = misses[:limit]
    details = [
        {
            "rank": rank,
            "file_name": image.file_name,
            "box_xyxy": [round(value, 1) for value in (box.x1, box.y1, box.x2, box.y2)],
            "width_px": round(box.width, 1),
        }
        for rank, (image, box) in enumerate(chosen, start=1)
    ]
    if not chosen:
        return details
    rows = (len(chosen) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image_result, box) in enumerate(chosen):
        with Image.open(dataset_dir / image_result.file_name) as source:
            source = source.convert("RGB")
            pad_x = (box.x2 - box.x1) * 0.5 + 8
            pad_y = (box.y2 - box.y1) * 0.5 + 8
            left, top = max(0, box.x1 - pad_x), max(0, box.y1 - pad_y)
            crop = source.crop(
                (left, top, min(source.width, box.x2 + pad_x), min(source.height, box.y2 + pad_y))
            )
        ImageDraw.Draw(crop).rectangle(
            [box.x1 - left, box.y1 - top, box.x2 - left, box.y2 - top], outline="red", width=2
        )
        crop.thumbnail((cell_width - 4, cell_height - 18))
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        sheet.paste(crop, (x + 2, y + 2))
        draw.text((x + 3, y + cell_height - 14), str(index + 1), fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)
    return details


def _split_dir(config: dict[str, Any], split: str) -> Path:
    """The holdout lives in its own folder (`holdout_dir`); other splits in `dataset_dir`."""
    if split == "holdout":
        if "holdout_dir" not in config:
            raise ValueError("Split 'holdout' needs holdout_dir in the config")
        return Path(config["holdout_dir"])
    return Path(config["dataset_dir"])


def evaluate(
    config: dict[str, Any],
    predictor: Predictor,
    run_dir: Path,
) -> dict[str, Any]:
    """Run `predictor` on each configured split and write predictions, metrics and galleries."""
    class_map = {int(k): int(v) for k, v in (config["model"].get("class_map") or {}).items()}
    model_classes = frozenset(class_map.values()) if class_map else frozenset(CLASS_NAMES)
    metrics: dict[str, Any] = {}
    metrics_all_plates: dict[str, Any] = {}
    metrics_audited: dict[str, Any] = {}
    worst_misses: dict[str, Any] = {}
    for split in config["splits"]:
        if split not in EVAL_SPLITS:
            raise ValueError(f"Unknown split {split!r}")
        dataset_dir = _split_dir(config, split)
        images = load_split(dataset_dir, split)
        LOGGER.info("Evaluating %s: %d images", split, len(images))
        with (run_dir / f"predictions_{split}.jsonl").open("w", encoding="utf-8") as stream:
            for number, image_result in enumerate(images, start=1):
                with Image.open(dataset_dir / image_result.file_name) as image:
                    image_result.detections = predictor(image)
                stream.write(
                    json.dumps(
                        {
                            "file_name": image_result.file_name,
                            "detections": [
                                [
                                    round(d.x1, 2),
                                    round(d.y1, 2),
                                    round(d.x2, 2),
                                    round(d.y2, 2),
                                    round(d.score, 5),
                                    d.class_id,
                                ]
                                for d in image_result.detections
                            ],
                        }
                    )
                    + "\n"
                )
                if number % 250 == 0:
                    LOGGER.info("  %s: %d/%d", split, number, len(images))
        min_plate_width = config.get("min_plate_width")
        metrics[split] = summarize(
            images,
            plate_conf=float(config["plate_conf"]),
            vehicle_conf=float(config["vehicle_conf"]),
            model_classes=model_classes,
            min_plate_width=min_plate_width,
        )
        audit_ids = (config.get("audit_ignored_ids") or {}).get(split)
        if audit_ids is not None:
            metrics_audited[split] = summarize(
                images,
                plate_conf=float(config["plate_conf"]),
                vehicle_conf=float(config["vehicle_conf"]),
                model_classes=model_classes,
                min_plate_width=min_plate_width,
                ignored_ids=audit_ids,
            )
        if min_plate_width is not None:
            metrics_all_plates[split] = summarize(
                images,
                plate_conf=float(config["plate_conf"]),
                vehicle_conf=float(config["vehicle_conf"]),
                model_classes=model_classes,
            )
        worst_misses[split] = {}
        thresholds = {1: float(config["plate_conf"]), 2: float(config["vehicle_conf"])}
        for class_id in sorted(model_classes):
            name = CLASS_NAMES[class_id]
            worst_misses[split][name] = write_gallery(
                missed_boxes(
                    images,
                    class_id,
                    thresholds[class_id],
                    min_plate_width=config.get("min_plate_width"),
                ),
                dataset_dir,
                run_dir / f"misses_{split}_{name}.jpg",
                int(config.get("gallery_size", 30)),
            )
    results: dict[str, Any] = {"metrics": metrics, "worst_misses": worst_misses}
    if metrics_all_plates:
        results["metrics_all_plates"] = metrics_all_plates
    if metrics_audited:
        results["metrics_audited"] = metrics_audited
    return results


def run_baseline(
    config_path: Path,
    *,
    runs_dir: Path = Path("runs"),
    predictor: Predictor | None = None,
) -> Path:
    """Evaluate the configured model; returns the run directory."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Baseline config must be a YAML mapping")
    model = config["model"]
    manifest_hashes = {
        str(directory): sha256_file(directory / "manifest.csv")
        for directory in {_split_dir(config, split) for split in config["splits"]}
    }
    if predictor is None:
        model_path = _prepare_model(model)
        predictor = OnnxYoloxPredictor(
            model_path,
            input_size=int(model["input_size"]),
            channel_order=str(model.get("channel_order", "rgb")),
            conf_threshold=float(config.get("collect_conf", 0.01)),
            nms_iou=float(config.get("nms_iou", DEFAULT_NMS_IOU)),
            class_map={int(k): int(v) for k, v in model["class_map"].items()},
        )
        checkpoint_hash = sha256_file(model_path)
    else:
        checkpoint_hash = model.get("sha256", "injected-predictor")
    run_id = f"baseline-{config['name']}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    shutil.copy(config_path, run_dir / "config.yaml")
    results = evaluate(config, predictor, run_dir)
    summary = {
        "run_id": run_id,
        "kind": "baseline",
        "git": _git_state(),
        "config": config,
        "dataset_manifest_sha256": manifest_hashes,
        "checkpoint_sha256": checkpoint_hash,
        "model_url": model.get("url"),
        **results,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Baseline written to %s", run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(run_baseline(args.config))


if __name__ == "__main__":
    main()

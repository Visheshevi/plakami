"""Vehicle pseudo-labels for plate-only training images (ADR-0011).

CCPD and Kaggle images carry plate boxes only. YOLOX's objectness is shared by both
classes, so their unlabelled vehicles would be learned as background. This step runs an
Apache-2.0 COCO detector (official YOLOX-L) on those images and keeps confident vehicle
boxes. The output is used for *training labels only* (train and val training files); the
evaluation splits keep their original labels and `annotated_classes` scoping.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from platedet.eval.baseline import (
    OnnxYoloxPredictor,
    Predictor,
    _prepare_model,
    load_split,
    sha256_file,
)

LOGGER = logging.getLogger("platedet.train")
VEHICLE = 2
PSEUDO_SPLITS = ("train", "val")


def pseudo_label_split(
    dataset_dir: Path, split: str, predictor: Predictor, score_threshold: float
) -> dict[str, list[list[float]]]:
    """Confident vehicle boxes (x1, y1, x2, y2, score) for images that lack vehicle labels."""
    labels: dict[str, list[list[float]]] = {}
    images = [
        image for image in load_split(dataset_dir, split) if VEHICLE not in image.annotated_classes
    ]
    for number, image_result in enumerate(images, start=1):
        with Image.open(dataset_dir / image_result.file_name) as image:
            detections = predictor(image)
        labels[image_result.file_name] = [
            [round(d.x1, 2), round(d.y1, 2), round(d.x2, 2), round(d.y2, 2), round(d.score, 4)]
            for d in detections
            if d.class_id == VEHICLE and d.score >= score_threshold
        ]
        if number % 250 == 0:
            LOGGER.info("  %s: %d/%d", split, number, len(images))
    return labels


def build_pseudo_labels(config: dict[str, Any], predictor: Predictor | None = None) -> Path:
    model = config["model"]
    if predictor is None:
        predictor = OnnxYoloxPredictor(
            _prepare_model(model),
            input_size=int(model["input_size"]),
            channel_order=str(model["channel_order"]),
            conf_threshold=float(config["score_threshold"]),
            nms_iou=float(config["nms_iou"]),
            class_map={int(k): int(v) for k, v in model["class_map"].items()},
        )
    dataset_dir = Path(config["dataset_dir"])
    threshold = float(config["score_threshold"])
    labels: dict[str, list[list[float]]] = {}
    for split in config.get("splits", PSEUDO_SPLITS):
        if split not in PSEUDO_SPLITS:
            raise ValueError(f"Pseudo-labels are for training splits only, not {split!r}")
        labels.update(pseudo_label_split(dataset_dir, split, predictor, threshold))
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "model_url": model["url"],
        "model_sha256": model["sha256"],
        "score_threshold": threshold,
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.csv"),
        "images": len(labels),
        "boxes": sum(len(boxes) for boxes in labels.values()),
        "labels": labels,
    }
    output.write_text(json.dumps(result) + "\n", encoding="utf-8")
    LOGGER.info("Pseudo-labels: %d images, %d vehicle boxes", result["images"], result["boxes"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(build_pseudo_labels(yaml.safe_load(args.config.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()

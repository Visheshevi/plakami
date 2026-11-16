"""Run the release candidate on your own photos and save copies with the boxes drawn.

`make try` (photos in data/my_photos/) or `make try IMAGES=~/Pictures/cars`
(also `OUT=…`, `MODEL=onnx`, `NO_TILES=1`)

Uses the shipped model (TFLite FP16 by default), the recommended thresholds and 2x2 tiling
from `configs/release/v1.yaml`, and the shared pre/post-processing. Phone photos are turned
upright using their EXIF orientation first. Writes, per photo, `<name>_boxes.jpg` with plates
in red and vehicles in blue, plus `detections.csv` with every box, to the output folder.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageOps

from platedet.parity.parity import Runner
from platedet.postprocess import Detection, detect_tiled

RELEASE_CONFIG = Path("configs/release/v1.yaml")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}
COLOURS = {1: (255, 40, 40), 2: (40, 120, 255)}


def load_release(path: Path = RELEASE_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def find_images(inputs: list[Path]) -> list[Path]:
    found: list[Path] = []
    for item in inputs:
        if item.is_dir():
            found += sorted(p for p in item.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        elif item.suffix.lower() in IMAGE_SUFFIXES:
            found.append(item)
    return found


def keep(detections: list[Detection], thresholds: dict[int, float]) -> list[Detection]:
    return [d for d in detections if d.score >= thresholds.get(d.class_id, 1.0)]


def draw(image: Image.Image, detections: list[Detection], labels: list[str]) -> Image.Image:
    canvas = image.copy()
    pen = ImageDraw.Draw(canvas)
    width = max(2, round(max(image.size) / 400))
    for d in sorted(detections, key=lambda d: d.class_id, reverse=True):  # plates on top
        colour = COLOURS.get(d.class_id, (255, 255, 0))
        pen.rectangle([d.x1, d.y1, d.x2, d.y2], outline=colour, width=width)
        label = f"{labels[d.class_id - 1]} {d.score:.2f}"
        pen.text((d.x1 + width, max(0, d.y1 - 12 - width)), label, fill=colour)
    return canvas


def try_model(
    inputs: list[Path],
    out_dir: Path,
    *,
    model: str = "tflite_fp16",
    tiles: bool = True,
    release: dict[str, Any] | None = None,
    runs_dir: Path = Path("runs"),
) -> dict[str, Any]:
    release = release or load_release()
    run_dir = runs_dir / release["run_id"]
    runner = Runner(
        run_dir / release["model_files"][model],
        int(release["input_size"]),
        str(release["channel_order"]),
        float(release["nms_iou"]),
    )
    labels = list(release["labels"])
    thresholds = {
        index + 1: float(release["recommended_thresholds"][name])
        for index, name in enumerate(labels)
    }
    tiling = release["tiling"]
    images = find_images(inputs)
    if not images:
        raise FileNotFoundError(f"No images found in {', '.join(map(str, inputs))}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    seconds: list[float] = []
    for path in images:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        start = time.perf_counter()
        if tiles:
            detections = detect_tiled(
                image,
                runner,
                grid=int(tiling["grid"]),
                overlap=float(tiling["overlap"]),
                nms_iou=float(release["nms_iou"]),
            )
        else:
            detections = runner(image)
        seconds.append(time.perf_counter() - start)
        detections = keep(detections, thresholds)
        draw(image, detections, labels).save(out_dir / f"{path.stem}_boxes.jpg", quality=90)
        for d in detections:
            rows.append(
                {
                    "image": str(path),
                    "class": labels[d.class_id - 1],
                    "score": round(d.score, 4),
                    "x1": round(d.x1, 1),
                    "y1": round(d.y1, 1),
                    "x2": round(d.x2, 1),
                    "y2": round(d.y2, 1),
                }
            )
        plates = sum(d.class_id == 1 for d in detections)
        vehicles = sum(d.class_id == 2 for d in detections)
        print(f"{path.name}: {plates} plate(s), {vehicles} vehicle(s), {seconds[-1] * 1000:.0f} ms")

    with (out_dir / "detections.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["image", "class", "score", "x1", "y1", "x2", "y2"]
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "images": len(images),
        "plates": sum(r["class"] == labels[0] for r in rows),
        "vehicles": sum(r["class"] == labels[1] for r in rows),
        "median_ms": round(sorted(seconds)[len(seconds) // 2] * 1000),
        "out_dir": str(out_dir),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", choices=("tflite_fp16", "onnx"), default="tflite_fp16")
    parser.add_argument("--no-tiles", action="store_true")
    args = parser.parse_args()
    out = args.out or Path("runs/try") / datetime.now().strftime("%Y%m%d-%H%M%S")
    summary = try_model(args.images, out, model=args.model, tiles=not args.no_tiles)
    print(
        f"\n{summary['images']} photos: {summary['plates']} plates, {summary['vehicles']} "
        f"vehicles, median {summary['median_ms']} ms per photo. Results in {summary['out_dir']}/"
    )


if __name__ == "__main__":
    main()

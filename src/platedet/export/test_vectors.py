"""Test vectors for porting the pre/post-processing (SPEC §8). `make test-vectors`.

Writes `runs/<run_id>/release/test_vectors/` with, per vector:

  input.png          the photo, losslessly re-encoded (so JPEG decoders can't differ)
  letterboxed.png    the 640x640 model input canvas as 8-bit pixels, in RGB order
  letterbox.json     ratio, resized size, tensor shape and SHA-256 of the float32 tensor
  raw_output.npy     ONNX FP32 head output (1, N, 7) for that tensor
  expected.json      detections at the recommended thresholds: single pass and 2x2 tiles,
                     from ONNX FP32 (exact reference) and TFLite FP16 (for phone ports)

plus `vectors.json` (index and tolerances) and `ATTRIBUTION.md` for the photos.
`--check` re-derives every expected file with the shipped code and fails on any difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from platedet.parity.parity import Runner
from platedet.postprocess import Detection, detect_tiled, letterbox, postprocess
from platedet.tools.try_model import load_release

DATASET = Path("data/processed/combined")
# (vector name, file in the combined dataset). Chosen by sha256("vectors:" + file_name)
# within each source and orientation among test images with a plate.
PHOTOS = (
    (
        "ccpd_portrait",
        "images/ccpd/CCPD2020/ccpd_green/test/038553639846743294-90_262-179&511_529&611-"
        "529&611_191&611_179&516_525&511-0_0_3_25_27_31_26_31-111-39.jpg",
    ),
    ("kaggle_small", "images/kaggle_car_plate_detection/Cars246.png"),
    ("open_images_landscape", "images/open_images_v6/4289dcda09d077ed.jpg"),
    ("open_images_portrait", "images/open_images_v6/b46460a14906f889.jpg"),
)
BLANK = ("blank_grey", (800, 600), (128, 128, 128))
TOLERANCES = {
    "letterboxed_pixels": "within 1 grey level of letterboxed.png (bilinear rounding)",
    "onnx_boxes": "same boxes and classes; coordinates within 0.5 px, scores within 0.001",
    "fp16_boxes": "same plates; coordinates within 2 px, scores within 0.05 (SPEC §7)",
}


def _detections(items: list[Detection], labels: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "class": labels[d.class_id - 1],
            "score": round(d.score, 5),
            "box": [round(v, 2) for v in (d.x1, d.y1, d.x2, d.y2)],
        }
        for d in items
    ]


def _keep(items: list[Detection], thresholds: dict[int, float]) -> list[Detection]:
    return [d for d in items if d.score >= thresholds[d.class_id]]


def build_vector(
    name: str,
    image: Image.Image,
    runners: dict[str, Runner],
    release: dict[str, Any],
) -> dict[str, Any]:
    """All expected values for one image (no files written)."""
    size = int(release["input_size"])
    labels = list(release["labels"])
    thresholds = {i + 1: float(release["recommended_thresholds"][n]) for i, n in enumerate(labels)}
    tensor, ratio = letterbox(image, size, str(release["channel_order"]))
    raw = runners["onnx"].raw(tensor)
    single = postprocess(
        raw,
        size=size,
        ratio=ratio,
        image_width=image.width,
        image_height=image.height,
        conf_threshold=min(thresholds.values()),
        nms_iou=float(release["nms_iou"]),
    )
    expected: dict[str, Any] = {"name": name, "image_size": [image.width, image.height]}
    for kind, runner in runners.items():
        tiled = detect_tiled(
            image,
            runner,
            grid=int(release["tiling"]["grid"]),
            overlap=float(release["tiling"]["overlap"]),
            nms_iou=float(release["nms_iou"]),
        )
        expected[kind] = {
            "single": _detections(_keep(runner(image), thresholds), labels),
            "tiles2": _detections(_keep(tiled, thresholds), labels),
        }
    assert _detections(_keep(single, thresholds), labels) == expected["onnx"]["single"]
    canvas = tensor[0].transpose(1, 2, 0).round().astype(np.uint8)
    if release["channel_order"] == "bgr":
        canvas = canvas[:, :, ::-1]
    return {
        "expected": expected,
        "letterbox": {
            "ratio": ratio,
            "resized": [int(image.width * ratio), int(image.height * ratio)],
            "tensor_shape": list(tensor.shape),
            "channel_order": release["channel_order"],
            "tensor_sha256": hashlib.sha256(tensor.astype(np.float32).tobytes()).hexdigest(),
        },
        "canvas": canvas,
        "raw": raw.astype(np.float32),
    }


def _inputs(dataset: Path) -> list[tuple[str, Image.Image, dict[str, str]]]:
    import csv

    with (dataset / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = {row["source_image_id"]: row for row in csv.DictReader(stream)}
    items = []
    for name, file_name in PHOTOS:
        row = rows[file_name.removeprefix("images/")]
        with Image.open(dataset / file_name) as source:
            items.append((name, source.convert("RGB"), row))
    name, size, colour = BLANK
    items.append((name, Image.new("RGB", size, colour), {}))
    return items


def generate(out_dir: Path, *, dataset: Path = DATASET, runs_dir: Path = Path("runs")) -> Path:
    release = load_release()
    run_dir = runs_dir / release["run_id"]
    runners = {
        "onnx": Runner(run_dir / release["model_files"]["onnx"], 640, "bgr", 0.45),
        "fp16": Runner(run_dir / release["model_files"]["tflite_fp16"], 640, "bgr", 0.45),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    index, credits = [], []
    for name, image, row in _inputs(dataset):
        folder = out_dir / name
        folder.mkdir(exist_ok=True)
        image.save(folder / "input.png")
        built = build_vector(name, image, runners, release)
        Image.fromarray(built["canvas"]).save(folder / "letterboxed.png")
        (folder / "letterbox.json").write_text(json.dumps(built["letterbox"], indent=2) + "\n")
        np.save(folder / "raw_output.npy", built["raw"])
        (folder / "expected.json").write_text(json.dumps(built["expected"], indent=2) + "\n")
        index.append({"name": name, "source": row.get("source", "synthetic")})
        if row:
            credits.append(
                f"- `{name}/input.png`: {row['url']} by {row['author']}, {row['licence']} "
                f"({row['licence_url']}). Re-encoded as PNG; not otherwise modified."
            )
    (out_dir / "vectors.json").write_text(
        json.dumps(
            {
                "model_run": release["run_id"],
                "release": release,
                "tolerances": TOLERANCES,
                "vectors": index,
            },
            indent=2,
        )
        + "\n"
    )
    (out_dir / "ATTRIBUTION.md").write_text(
        "# Test vector photos\n\n" + "\n".join(credits) + "\n\n`blank_grey` is synthetic.\n"
    )
    return out_dir


def check(out_dir: Path, *, runs_dir: Path = Path("runs")) -> list[str]:
    """Re-derive every vector from input.png with the shipped code; list differences."""
    release = load_release()
    run_dir = runs_dir / release["run_id"]
    runners = {
        "onnx": Runner(run_dir / release["model_files"]["onnx"], 640, "bgr", 0.45),
        "fp16": Runner(run_dir / release["model_files"]["tflite_fp16"], 640, "bgr", 0.45),
    }
    problems = []
    for entry in json.loads((out_dir / "vectors.json").read_text())["vectors"]:
        folder = out_dir / entry["name"]
        with Image.open(folder / "input.png") as source:
            built = build_vector(entry["name"], source.convert("RGB"), runners, release)
        if built["letterbox"] != json.loads((folder / "letterbox.json").read_text()):
            problems.append(f"{entry['name']}: letterbox differs")
        if built["expected"] != json.loads((folder / "expected.json").read_text()):
            problems.append(f"{entry['name']}: detections differ")
        if not np.array_equal(built["raw"], np.load(folder / "raw_output.npy")):
            problems.append(f"{entry['name']}: raw output differs")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    out = Path("runs") / load_release()["run_id"] / "release" / "test_vectors"
    if args.check:
        problems = check(out)
        print("\n".join(problems) or f"All test vectors in {out} reproduce.")
        raise SystemExit(1 if problems else 0)
    print(generate(out))


if __name__ == "__main__":
    main()

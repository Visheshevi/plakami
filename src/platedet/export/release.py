"""Build and check the release package (SPEC §8, ADR-0018). `make release`, `make release-check`.

`make release` writes `runs/<run_id>/release/v<version>/` and a zip next to it:

  detector.onnx, detector_fp16.tflite   model files (INT8 not shipped, ADR-0016)
  labels.txt, release.json              classes; pre/post-processing contract, thresholds,
                                        metrics, parity, provenance and file hashes
  postprocess.py, test_vectors/         reference code and expected outputs for porting
  README.md, MODEL_CARD.md              integration guide; intended use, metrics and limits
  LICENSE, THIRD_PARTY_NOTICES.md,      MIT for this release; YOLOX (Apache-2.0) and
  licenses/, ATTRIBUTION.md             training-data credits
  SHA256SUMS                            `shasum -a 256 -c SHA256SUMS` checks every file

The builder refuses to package model files that differ from the ones evaluated and
parity-checked. `make release-check` copies the package to an empty folder, verifies the
checksums and reproduces every test vector using only the packaged files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import string
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from platedet.eval.baseline import _git_state, apply_box_corrections, load_split, sha256_file
from platedet.eval.label_audit import corrected_boxes, flagged_annotation_ids
from platedet.eval.metrics import READABLE_PLATE_WIDTH, summarize
from platedet.postprocess import TILE_EDGE_MARGIN, Detection
from platedet.tools.try_model import RELEASE_CONFIG, load_release

ROOT = Path(__file__).resolve().parents[3]
SPLIT_DIRS = {"test": Path("data/processed/combined"), "holdout": Path("data/processed/holdout")}
TRAINING_ATTRIBUTION = Path("data/processed/combined/ATTRIBUTION.md")
COPYRIGHT_PLACEHOLDER = "[COPYRIGHT HOLDER]"
SHIPPED = {"onnx": "detector.onnx", "tflite_fp16": "detector_fp16.tflite"}
OUTPUT_COLUMNS = ["cx", "cy", "w", "h", "objectness"]


# --- checksums -------------------------------------------------------------------------


def write_sha256sums(package: Path) -> Path:
    """`SHA256SUMS` in `shasum -a 256` format for every file except itself, sorted."""
    lines = [
        f"{sha256_file(path)}  {path.relative_to(package).as_posix()}"
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    target = package / "SHA256SUMS"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def verify_sha256sums(package: Path) -> list[str]:
    """Problems found: changed, missing or unlisted files."""
    problems = []
    listed = set()
    for line in (package / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        listed.add(name)
        path = package / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif sha256_file(path) != digest:
            problems.append(f"checksum differs: {name}")
    for path in package.rglob("*"):
        name = path.relative_to(package).as_posix()
        if path.is_file() and name != "SHA256SUMS" and name not in listed:
            problems.append(f"not in SHA256SUMS: {name}")
    return problems


# --- metrics at the recommended thresholds ---------------------------------------------


def _with_predictions(split: str, split_dir: Path, eval_dir: Path) -> list[Any]:
    images = load_split(split_dir, split)
    saved = {}
    for line in (eval_dir / f"predictions_{split}.jsonl").read_text("utf-8").splitlines():
        record = json.loads(line)
        saved[record["file_name"]] = [
            Detection(*map(float, d[:5]), int(d[5])) for d in record["detections"]
        ]
    for image in images:
        image.detections = saved[image.file_name]
    return images


def operating_metrics(eval_dir: Path, thresholds: dict[str, float]) -> dict[str, Any]:
    """Audited test/holdout metrics at the recommended thresholds, from saved predictions.

    Reporting only: the thresholds were chosen on val (ADR-0017).
    """
    metrics = {}
    for split, split_dir in SPLIT_DIRS.items():
        ignored = flagged_annotation_ids(split, split_dir)
        images = _with_predictions(split, split_dir, eval_dir)
        if ignored is not None:
            images = apply_box_corrections(images, corrected_boxes(split))
        metrics[split] = summarize(
            images,
            plate_conf=thresholds["license_plate"],
            vehicle_conf=thresholds["vehicle"],
            min_plate_width=READABLE_PLATE_WIDTH,
            ignored_ids=ignored or frozenset(),
        )
    return metrics


# --- release.json and the model card ---------------------------------------------------


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v) for v in value]
    return value


def _class_metrics(split_metrics: dict[str, Any]) -> dict[str, Any]:
    plate, vehicle = split_metrics["license_plate"], split_metrics["vehicle"]
    return _round(
        {
            "images": split_metrics["images"],
            "plate_recall": plate["recall"],
            "small_plate_recall": plate["small_plates"]["recall"],
            "plate_false_positives_per_image": plate["false_positives_per_image"],
            "plate_ap50": plate["ap50"],
            "plates": plate["ground_truth"],
            "vehicle_recall": vehicle.get("recall"),
            "vehicle_false_positives_per_image": vehicle.get("false_positives_per_image"),
            "vehicle_ap50": vehicle.get("ap50"),
        }
    )


def release_manifest(
    release: dict[str, Any],
    run_dir: Path,
    package: Path,
    summary: dict[str, Any],
    parity: dict[str, Any],
    at_recommended: dict[str, Any],
) -> dict[str, Any]:
    run_info = json.loads((run_dir / "run_info.json").read_text("utf-8"))
    export = json.loads((run_dir / "export.json").read_text("utf-8"))
    labels = list(release["labels"])
    size = int(release["input_size"])
    cells = sum((size // s) ** 2 for s in (8, 16, 32))
    return {
        "name": "platedet",
        "version": release["version"],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "license": "MIT",
        "labels": labels,
        "files": {
            name: {"sha256": sha256_file(package / name), "bytes": (package / name).stat().st_size}
            for name in SHIPPED.values()
        },
        "input": {
            "size": size,
            "channel_order": release["channel_order"],
            "value_range": [0, 255],
            "normalisation": "none (raw pixel values as float32)",
            "resize": "keep aspect ratio; ratio = min(size/width, size/height); new size = "
            "int(width*ratio) x int(height*ratio); bilinear, OpenCV INTER_LINEAR semantics",
            "placement": "top-left of the size x size canvas",
            "pad_value": 114,
            "orientation": "apply EXIF orientation first (photo upright as the user sees it)",
            "onnx": {"name": "images", "shape": [1, 3, size, size], "layout": "NCHW"},
            "tflite": {"name": "images", "shape": [1, size, size, 3], "layout": "NHWC"},
        },
        "output": {
            "name": "output",
            "shape": [1, cells, 5 + len(labels)],
            "columns": OUTPUT_COLUMNS + labels,
            "strides": [8, 16, 32],
            "row_order": "stride 8 grid, then 16, then 32; row-major (y, then x) in each",
            "decoded_in_model": False,
            "box_decoding": "cx = (raw_cx + grid_x) * stride; cy = (raw_cy + grid_y) * stride; "
            "w = exp(raw_w) * stride; h = exp(raw_h) * stride (input pixels)",
            "scores": "objectness and class columns are probabilities already (sigmoid applied); "
            "score = objectness * best class probability; class = argmax",
        },
        "postprocess": {
            "nms": "class-aware greedy NMS",
            "nms_iou": float(release["nms_iou"]),
            "to_original_pixels": "divide by ratio, clip to the image",
        },
        "recommended_thresholds": release["recommended_thresholds"],
        "tiling": {
            **release["tiling"],
            "edge_margin": TILE_EDGE_MARGIN,
            "edge_margin_rule": "drop a tile box within edge_margin * tile size (at least 2 px) "
            "of an inner tile edge, then merge all passes with class-aware NMS",
            "recommended": "single photos (ADR-0013); about 5 model passes per photo",
        },
        "metrics": {
            "note": "ONNX FP32 with 2x2 tiles; test and holdout labels audited (ADR-0014); "
            f"plates under {READABLE_PLATE_WIDTH:g} px wide ignored (ADR-0013)",
            "at_recommended_thresholds": {s: _class_metrics(m) for s, m in at_recommended.items()},
            "spec_targets": [
                _round({k: row[k] for k in ("metric", "value", "target", "met")})
                for row in summary["targets_audited"]
            ],
            "accepted_gap": "owner accepted the gap to the SPEC §6 targets (ADR-0015)",
        },
        "parity": {
            "reference": "ONNX FP32",
            "images": parity["images"],
            "criteria": parity["criteria"],
            "fp16": {
                mode: {
                    "passed": result["passed"],
                    **{
                        cls: {
                            k: result[cls][k]
                            for k in ("reference_boxes", "off_tolerance", "missing")
                        }
                        for cls in labels
                    },
                }
                for mode, result in parity["results"]["fp16"].items()
                if isinstance(result, dict)
            },
            "speed_ms_mac_cpu": parity["speed"],
        },
        "provenance": {
            "release_commit": _git_state(),
            "training_commit": run_info["git"],
            "run_id": run_dir.name,
            "checkpoint_sha256": export["checkpoint_sha256"],
            "pretrained": run_info["config"]["pretrained"],
            "dataset_manifest_sha256": summary["manifest_sha256"],
            "evaluation_commit": summary["git"],
            "onnx_opset": export["opset"],
            "tflite_converter": f"onnx2tf {export['tflite']['onnx2tf']} "
            f"({export['tflite']['backend']})",
        },
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_model_card(template: str, manifest: dict[str, Any]) -> str:
    """Fill `docs/release/MODEL_CARD.md.tmpl` ($placeholders). Fails on any left unfilled."""
    rec = manifest["metrics"]["at_recommended_thresholds"]
    rows = "\n".join(
        f"| {row['metric']} | {_fmt(row['value'])} | {row['target']} | "
        f"{'yes' if row['met'] else 'no'} |"
        for row in manifest["metrics"]["spec_targets"]
    )
    speed = manifest["parity"]["speed_ms_mac_cpu"]
    single = manifest["parity"]["fp16"]["single"]
    tiles = manifest["parity"]["fp16"]["tiles2"]
    values = {
        "version": manifest["version"],
        "plate_threshold": f"{manifest['recommended_thresholds']['license_plate']:.2f}",
        "vehicle_threshold": f"{manifest['recommended_thresholds']['vehicle']:.2f}",
        "spec_rows": rows,
        "parity_images": f"{manifest['parity']['images']:,}",
        "single_boxes": f"{sum(single[c]['reference_boxes'] for c in manifest['labels']):,}",
        "tiles_boxes": f"{sum(tiles[c]['reference_boxes'] for c in manifest['labels']):,}",
        "tiles_off": str(sum(tiles[c]["off_tolerance"] for c in manifest["labels"])),
        "tiles_missing": str(sum(tiles[c]["missing"] for c in manifest["labels"])),
        "onnx_ms": f"{speed['onnx']['median_ms_per_photo_2x2_tiles']:.0f}",
        "fp16_ms": f"{speed['fp16']['median_ms_per_photo_2x2_tiles']:.0f}",
        "onnx_pass_ms": f"{speed['onnx']['median_ms_per_pass']:.0f}",
        "fp16_pass_ms": f"{speed['fp16']['median_ms_per_pass']:.0f}",
        "onnx_mb": f"{manifest['files']['detector.onnx']['bytes'] / 1e6:.1f}",
        "fp16_mb": f"{manifest['files']['detector_fp16.tflite']['bytes'] / 1e6:.1f}",
        "checkpoint_sha": manifest["provenance"]["checkpoint_sha256"],
        "commit": manifest["provenance"]["release_commit"]["commit"],
    }
    for split in ("test", "holdout"):
        m = rec[split]
        values |= {
            f"{split}_images": f"{m['images']:,}",
            f"{split}_plates": f"{m['plates']:,}",
            f"{split}_plate_recall": _fmt(m["plate_recall"]),
            f"{split}_small": _fmt(m["small_plate_recall"]),
            f"{split}_small_missed": f"{100 * (1 - m['small_plate_recall']):.0f}",
            f"{split}_plate_fp": _fmt(m["plate_false_positives_per_image"], 2),
            f"{split}_vehicle_recall": _fmt(m["vehicle_recall"]),
            f"{split}_vehicle_fp": _fmt(m["vehicle_false_positives_per_image"], 2),
        }
    return string.Template(template).substitute(values)


# --- licences and attribution ----------------------------------------------------------


def training_attribution(source: Path) -> str:
    body = source.read_text(encoding="utf-8").split("\n", 1)[1].lstrip("\n")
    return (
        "# Attribution\n\n"
        "The model weights were trained on the images below (train split; val was used to\n"
        "pick checkpoints and thresholds, test only to measure). No images are included in\n"
        "this release except the test-vector photos, which are credited in\n"
        "`test_vectors/ATTRIBUTION.md`.\n\n"
        "The weights start from the YOLOX-Tiny COCO checkpoint by Megvii (Apache-2.0); see\n"
        "`THIRD_PARTY_NOTICES.md`.\n\n" + body
    )


# --- build -----------------------------------------------------------------------------


def _check_same(path: Path, expected: str, what: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{path} ({actual[:12]}) is not the file {what} ({expected[:12]})")


def build(
    *,
    release_config: Path = RELEASE_CONFIG,
    runs_dir: Path = Path("runs"),
    root: Path = ROOT,
) -> Path:
    release = load_release(release_config)
    run_dir = runs_dir / release["run_id"]
    summary = json.loads((run_dir / "eval_tiles2" / "summary.json").read_text("utf-8"))
    parity = json.loads((run_dir / "parity" / "parity.json").read_text("utf-8"))
    onnx = run_dir / release["model_files"]["onnx"]
    fp16 = run_dir / release["model_files"]["tflite_fp16"]
    _check_same(onnx, summary["model_sha256"], "evaluated")
    _check_same(onnx, parity["sha256"]["onnx"], "parity-checked")
    _check_same(fp16, parity["sha256"]["fp16"], "parity-checked")
    vectors = run_dir / "release" / "test_vectors"
    if json.loads((vectors / "vectors.json").read_text("utf-8"))["release"] != release:
        raise ValueError("Test vectors were built for a different release config; rerun them")

    package = run_dir / "release" / f"v{release['version']}"
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)
    shutil.copy2(onnx, package / SHIPPED["onnx"])
    shutil.copy2(fp16, package / SHIPPED["tflite_fp16"])
    (package / "labels.txt").write_text("\n".join(release["labels"]) + "\n", encoding="utf-8")
    shutil.copy2(root / "src" / "platedet" / "postprocess.py", package / "postprocess.py")
    shutil.copytree(vectors, package / "test_vectors")
    shutil.copy2(root / "README.md", package / "README.md")
    shutil.copy2(root / "LICENSE", package / "LICENSE")
    shutil.copy2(root / "THIRD_PARTY_NOTICES.md", package / "THIRD_PARTY_NOTICES.md")
    shutil.copytree(root / "licenses", package / "licenses")
    (package / "ATTRIBUTION.md").write_text(
        training_attribution(root / TRAINING_ATTRIBUTION), encoding="utf-8"
    )

    at_recommended = operating_metrics(run_dir / "eval_tiles2", release["recommended_thresholds"])
    manifest = release_manifest(release, run_dir, package, summary, parity, at_recommended)
    (package / "release.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    template = (root / "docs" / "release" / "MODEL_CARD.md.tmpl").read_text("utf-8")
    (package / "MODEL_CARD.md").write_text(render_model_card(template, manifest), "utf-8")
    write_sha256sums(package)

    archive = package.with_name(f"platedet-v{release['version']}.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                bundle.write(path, f"platedet-v{release['version']}/{path.relative_to(package)}")
    return package


# --- check (uses only the packaged files) ----------------------------------------------


def _load_packaged_postprocess(package: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "packaged_postprocess", package / "postprocess.py"
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["packaged_postprocess"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def boxes_match(expected: list[dict], actual: list[dict], px: float, score: float) -> bool:
    """Same classes in the same order, coordinates within `px`, scores within `score`."""
    if len(expected) != len(actual):
        return False
    return all(
        e["class"] == a["class"]
        and abs(e["score"] - a["score"]) <= score
        and all(abs(x - y) <= px for x, y in zip(e["box"], a["box"], strict=True))
        for e, a in zip(expected, actual, strict=True)
    )


def check(package: Path) -> list[str]:
    """Copy the package to an empty folder, verify checksums, reproduce every test vector."""
    import numpy as np
    from PIL import Image

    problems: list[str] = []
    with tempfile.TemporaryDirectory() as temporary:
        copy = Path(temporary) / package.name
        shutil.copytree(package, copy)
        problems += verify_sha256sums(copy)
        if COPYRIGHT_PLACEHOLDER in (copy / "LICENSE").read_text("utf-8"):
            problems.append(f"LICENSE still has the {COPYRIGHT_PLACEHOLDER} placeholder")
        manifest = json.loads((copy / "release.json").read_text("utf-8"))
        for name, info in manifest["files"].items():
            if sha256_file(copy / name) != info["sha256"]:
                problems.append(f"release.json hash differs: {name}")
        post = _load_packaged_postprocess(copy)
        size = manifest["input"]["size"]
        order = manifest["input"]["channel_order"]
        nms_iou = manifest["postprocess"]["nms_iou"]
        labels = manifest["labels"]
        thresholds = {i + 1: manifest["recommended_thresholds"][n] for i, n in enumerate(labels)}

        import onnxruntime
        from ai_edge_litert.interpreter import Interpreter

        session = onnxruntime.InferenceSession(
            str(copy / "detector.onnx"), providers=["CPUExecutionProvider"]
        )
        interpreter = Interpreter(model_path=str(copy / "detector_fp16.tflite"), num_threads=4)
        interpreter.allocate_tensors()
        tfl_in = interpreter.get_input_details()[0]["index"]
        tfl_out = interpreter.get_output_details()[0]["index"]

        def raw_onnx(tensor: Any) -> Any:
            return session.run(None, {"images": tensor})[0]

        def raw_fp16(tensor: Any) -> Any:
            interpreter.set_tensor(tfl_in, tensor.transpose(0, 2, 3, 1).copy())
            interpreter.invoke()
            return interpreter.get_tensor(tfl_out)

        def detector(raw_fn: Any) -> Any:
            def detect(image: Any) -> list[Any]:
                tensor, ratio = post.letterbox(image, size, order)
                return post.postprocess(
                    raw_fn(tensor),
                    size=size,
                    ratio=ratio,
                    image_width=image.width,
                    image_height=image.height,
                    conf_threshold=0.01,
                    nms_iou=nms_iou,
                )

            return detect

        def as_dicts(found: list[Any]) -> list[dict[str, Any]]:
            return [
                {
                    "class": labels[d.class_id - 1],
                    "score": round(d.score, 5),
                    "box": [round(v, 2) for v in (d.x1, d.y1, d.x2, d.y2)],
                }
                for d in found
                if d.score >= thresholds[d.class_id]
            ]

        tolerances = {"onnx": (0.5, 0.001), "fp16": (2.0, 0.05)}
        index = json.loads((copy / "test_vectors" / "vectors.json").read_text("utf-8"))
        for entry in index["vectors"]:
            folder = copy / "test_vectors" / entry["name"]
            with Image.open(folder / "input.png") as source:
                image = source.convert("RGB")
            tensor, _ = post.letterbox(image, size, order)
            letterbox_info = json.loads((folder / "letterbox.json").read_text("utf-8"))
            digest = hashlib.sha256(tensor.astype(np.float32).tobytes()).hexdigest()
            if digest != letterbox_info["tensor_sha256"]:
                problems.append(f"{entry['name']}: letterboxed tensor differs")
            if not np.allclose(raw_onnx(tensor), np.load(folder / "raw_output.npy"), atol=1e-3):
                problems.append(f"{entry['name']}: ONNX raw output differs")
            expected = json.loads((folder / "expected.json").read_text("utf-8"))
            for kind, raw_fn in (("onnx", raw_onnx), ("fp16", raw_fp16)):
                detect = detector(raw_fn)
                px, score = tolerances[kind]
                tiled = post.detect_tiled(
                    image,
                    detect,
                    grid=manifest["tiling"]["grid"],
                    overlap=manifest["tiling"]["overlap"],
                    nms_iou=nms_iou,
                )
                for mode, found in (("single", detect(image)), ("tiles2", tiled)):
                    if not boxes_match(expected[kind][mode], as_dicts(found), px, score):
                        problems.append(f"{entry['name']}: {kind} {mode} detections differ")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check the built package")
    args = parser.parse_args()
    release = yaml.safe_load(RELEASE_CONFIG.read_text("utf-8"))
    package = Path("runs") / release["run_id"] / "release" / f"v{release['version']}"
    if args.check:
        problems = check(package)
        print("\n".join(problems) or f"{package}: checksums OK; all test vectors reproduce.")
        raise SystemExit(1 if problems else 0)
    built = build()
    print(f"Release package: {built}\nArchive: {built.with_name(f'platedet-{built.name}.zip')}")


if __name__ == "__main__":
    main()

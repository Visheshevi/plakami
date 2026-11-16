"""ONNX vs TFLite parity, speed and size (SPEC §7, ADR-0016). `make parity RUN=<run_id>`.

For every test and holdout image, the ONNX FP32 model is the reference. Each TFLite file
runs on the same pre-processed input, and both go through the shared post-processing
(`platedet.postprocess`), once as a single pass and once with 2x2 tiles (ADR-0013).

Every reference box at or above its class's operating threshold (plates 0.2, vehicles 0.3)
is checked against the candidate's boxes of the same class (collected down to 0.01):

  within tolerance  best match has IoU >= 0.9 and |delta confidence| <= 0.05;
  missing           no box with IoU >= 0.5 and confidence >= threshold - 0.05;
  off               found, but outside the tolerance.

A file passes when no plate is missing and every reference box is within tolerance.
FP16 must pass; INT8 is published only if it also passes.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from platedet.eval.baseline import _git_state, load_split, sha256_file
from platedet.postprocess import (
    DEFAULT_TILE_OVERLAP,
    Detection,
    detect_tiled,
    iou_matrix,
    letterbox,
    postprocess,
)

LOGGER = logging.getLogger("platedet.parity")
THRESHOLDS = {1: 0.2, 2: 0.3}
CLASS_NAMES = {1: "license_plate", 2: "vehicle"}
MATCH_IOU = 0.9
CONF_TOLERANCE = 0.05
FOUND_IOU = 0.5
COLLECT_CONF = 0.01
TFLITE_THREADS = 4
SPLITS = {"test": Path("data/processed/combined"), "holdout": Path("data/processed/holdout")}


class Runner:
    """One model file: raw head outputs for an NCHW tensor, then shared post-processing."""

    def __init__(self, path: Path, size: int, channel_order: str, nms_iou: float) -> None:
        self.path = path
        self.size = size
        self.channel_order = channel_order
        self.nms_iou = nms_iou
        if path.suffix == ".onnx":
            import onnxruntime

            self.session = onnxruntime.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            self.kind = "onnx"
        else:
            from ai_edge_litert.interpreter import Interpreter

            self.interpreter = Interpreter(model_path=str(path), num_threads=TFLITE_THREADS)
            self.interpreter.allocate_tensors()
            self.input_index = self.interpreter.get_input_details()[0]["index"]
            self.output_index = self.interpreter.get_output_details()[0]["index"]
            self.kind = "tflite"

    def raw(self, tensor: np.ndarray) -> np.ndarray:
        if self.kind == "onnx":
            return self.session.run(None, {"images": tensor})[0]
        self.interpreter.set_tensor(self.input_index, tensor.transpose(0, 2, 3, 1).copy())
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)

    def __call__(self, image: Image.Image) -> list[Detection]:
        tensor, ratio = letterbox(image, self.size, self.channel_order)
        return postprocess(
            self.raw(tensor),
            size=self.size,
            ratio=ratio,
            image_width=image.width,
            image_height=image.height,
            conf_threshold=COLLECT_CONF,
            nms_iou=self.nms_iou,
        )


@dataclass
class Tally:
    reference_boxes: int = 0
    within: int = 0
    off: int = 0
    missing: int = 0
    max_conf_delta: float = 0.0
    min_iou: float = 1.0
    examples: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_boxes": self.reference_boxes,
            "within_tolerance": self.within,
            "off_tolerance": self.off,
            "missing": self.missing,
            "max_conf_delta": round(self.max_conf_delta, 4),
            "min_iou_of_found": round(self.min_iou, 4),
            "examples": self.examples[:10],
        }


def compare(
    reference: list[Detection],
    candidate: list[Detection],
    tallies: dict[int, Tally],
    file_name: str = "",
) -> None:
    """Add one image's comparison to the per-class tallies (see module docstring)."""
    for class_id, threshold in THRESHOLDS.items():
        tally = tallies[class_id]
        refs = [d for d in reference if d.class_id == class_id and d.score >= threshold]
        cands = [d for d in candidate if d.class_id == class_id]
        if not refs:
            continue
        ious = (
            iou_matrix(
                np.array([[d.x1, d.y1, d.x2, d.y2] for d in refs]),
                np.array([[d.x1, d.y1, d.x2, d.y2] for d in cands]),
            )
            if cands
            else np.zeros((len(refs), 0))
        )
        for index, ref in enumerate(refs):
            tally.reference_boxes += 1
            best = int(ious[index].argmax()) if cands else -1
            iou = float(ious[index, best]) if cands else 0.0
            score = cands[best].score if cands else 0.0
            delta = abs(score - ref.score)
            if iou < FOUND_IOU or score < threshold - CONF_TOLERANCE:
                tally.missing += 1
                kind = "missing"
            elif iou >= MATCH_IOU and delta <= CONF_TOLERANCE:
                tally.within += 1
                kind = ""
            else:
                tally.off += 1
                kind = "off"
            if iou >= FOUND_IOU:
                tally.max_conf_delta = max(tally.max_conf_delta, delta)
                tally.min_iou = min(tally.min_iou, iou)
            if kind:
                tally.examples.append(
                    {
                        "file_name": file_name,
                        "kind": kind,
                        "reference": [round(v, 1) for v in (ref.x1, ref.y1, ref.x2, ref.y2)],
                        "reference_score": round(ref.score, 4),
                        "iou": round(iou, 4),
                        "candidate_score": round(score, 4),
                    }
                )


def passed(tallies: dict[str, Any]) -> bool:
    plates, vehicles = tallies["license_plate"], tallies["vehicle"]
    return (
        plates["missing"] == 0
        and plates["off_tolerance"] == 0
        and vehicles["missing"] == 0
        and vehicles["off_tolerance"] == 0
    )


def _median_ms(samples: list[float]) -> float:
    return round(statistics.median(samples) * 1000, 2)


def measure_speed(
    runners: dict[str, Runner], images: list[Path], warmup: int = 5
) -> dict[str, Any]:
    """Median model latency per pass and per photo with 2x2 tiles, for each runner."""
    speed: dict[str, Any] = {}
    for name, runner in runners.items():
        passes, photos = [], []
        for index, path in enumerate(images):
            with Image.open(path) as source:
                image = source.convert("RGB")
            tensor, _ = letterbox(image, runner.size, runner.channel_order)
            start = time.perf_counter()
            runner.raw(tensor)
            passes.append(time.perf_counter() - start)
            start = time.perf_counter()
            detect_tiled(image, runner, grid=2, overlap=DEFAULT_TILE_OVERLAP)
            photos.append(time.perf_counter() - start)
            if index < warmup:
                passes.pop()
                photos.pop()
        speed[name] = {
            "median_ms_per_pass": _median_ms(passes),
            "median_ms_per_photo_2x2_tiles": _median_ms(photos),
            "images": len(passes),
            "threads": TFLITE_THREADS if runner.kind == "tflite" else "onnxruntime default",
        }
    return speed


def _detect(runner: Runner, image: Image.Image, mode: str, nms_iou: float) -> list[Detection]:
    if mode == "single":
        return runner(image)
    return detect_tiled(image, runner, grid=2, nms_iou=nms_iou)


def run_parity(
    run_dir: Path,
    *,
    splits: dict[str, Path] = SPLITS,
    speed_images: int = 50,
    limit: int | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    size = int(config["input_size"])
    channel_order = str(config.get("channel_order", "bgr"))
    nms_iou = float(config.get("nms_iou", 0.45))
    files = {"onnx": run_dir / "model.onnx"}
    for kind, name in (("fp16", "detector_fp16.tflite"), ("int8", "detector_int8.tflite")):
        if (run_dir / name).is_file():
            files[kind] = run_dir / name
    runners = {k: Runner(p, size, channel_order, nms_iou) for k, p in files.items()}
    candidates = [k for k in runners if k != "onnx"]
    if not candidates:
        raise FileNotFoundError(f"No TFLite files in {run_dir}; run make export first")

    modes = ("single", "tiles2")
    tallies = {(k, m): {c: Tally() for c in THRESHOLDS} for k in candidates for m in modes}
    image_paths: list[Path] = []
    for split, dataset_dir in splits.items():
        images = load_split(dataset_dir, split)[:limit]
        LOGGER.info("Parity on %s: %d images", split, len(images))
        for number, item in enumerate(images, start=1):
            path = dataset_dir / item.file_name
            image_paths.append(path)
            with Image.open(path) as source:
                image = source.convert("RGB")
            for mode in modes:
                reference = _detect(runners["onnx"], image, mode, nms_iou)
                for kind in candidates:
                    candidate = _detect(runners[kind], image, mode, nms_iou)
                    compare(reference, candidate, tallies[(kind, mode)], item.file_name)
            if number % 250 == 0:
                LOGGER.info("  %s: %d/%d", split, number, len(images))

    results: dict[str, Any] = {}
    for kind in candidates:
        results[kind] = {}
        for mode in modes:
            by_class = {CLASS_NAMES[c]: t.as_dict() for c, t in tallies[(kind, mode)].items()}
            results[kind][mode] = {"passed": passed(by_class), **by_class}
        results[kind]["passed"] = all(results[kind][m]["passed"] for m in modes)

    summary = {
        "run_id": run_dir.name,
        "git": _git_state(),
        "criteria": {
            "thresholds": {CLASS_NAMES[c]: t for c, t in THRESHOLDS.items()},
            "match_iou": MATCH_IOU,
            "conf_tolerance": CONF_TOLERANCE,
            "found_iou": FOUND_IOU,
        },
        "images": len(image_paths),
        "results": results,
        "speed": measure_speed(runners, image_paths[:speed_images]),
        "size_bytes": {k: p.stat().st_size for k, p in files.items()},
        "sha256": {k: sha256_file(p) for k, p in files.items()},
    }
    out = run_dir / "parity"
    out.mkdir(exist_ok=True)
    (out / "parity.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(out / "report.md", summary)
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [f"# Parity: {summary['run_id']}", "", f"{summary['images']} test + holdout images."]
    lines += [
        "",
        "| File | Mode | Class | Ref. boxes | Within | Off | Missing | Max Δconf | Min IoU |",
    ]
    lines += ["|---|---|---|---|---|---|---|---|---|"]
    for kind, by_mode in summary["results"].items():
        for mode in ("single", "tiles2"):
            for name in ("license_plate", "vehicle"):
                t = by_mode[mode][name]
                lines.append(
                    f"| {kind} | {mode} | {name} | {t['reference_boxes']} | "
                    f"{t['within_tolerance']} | {t['off_tolerance']} | {t['missing']} | "
                    f"{t['max_conf_delta']:.3f} | {t['min_iou_of_found']:.3f} |"
                )
        verdict = "**pass**" if by_mode["passed"] else "**fail**"
        lines.append(f"| {kind} | | | | | | | | {verdict} |")
    lines += ["", "## Speed (owner's Mac, CPU)", ""]
    lines += ["| File | ms per pass | ms per photo (2×2 tiles) | Threads |", "|---|---|---|---|"]
    for kind, s in summary["speed"].items():
        lines.append(
            f"| {kind} | {s['median_ms_per_pass']} | {s['median_ms_per_photo_2x2_tiles']} | "
            f"{s['threads']} |"
        )
    lines += ["", "## Size", "", "| File | MB |", "|---|---|"]
    for kind, size in summary["size_bytes"].items():
        lines.append(f"| {kind} | {size / 2**20:.2f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--limit", type=int, default=None, help="images per split (smoke runs)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run_parity(args.runs_dir / args.run, limit=args.limit)
    print(json.dumps({k: v["passed"] for k, v in summary["results"].items()}))


if __name__ == "__main__":
    main()

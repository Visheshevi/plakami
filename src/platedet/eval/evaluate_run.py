"""Evaluate a trained run's ONNX model on val, test and holdout (`make eval RUN=<run_id>`).

Fixed SPEC §6 thresholds, nothing tuned. Writes `runs/<run_id>/eval/` (or `eval_tiles<N>/`
with `--tiles N`) with summary.json, predictions, miss galleries and report.md (metrics
against the v1.0 targets and the worst misses with image paths).

Unreadable plates (under 12 px wide) are ignored (ADR-0013); the report also shows the
targets with every plate counted, as SPEC §6 is currently written, and, once the plate label
audit exists (`configs/audit/`, ADR-0014), with audit-flagged labels ignored.

`--rescore` reuses the saved predictions in the output folder instead of running the model.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from platedet.eval.baseline import (
    OnnxYoloxPredictor,
    TiledPredictor,
    _git_state,
    evaluate,
    sha256_file,
)
from platedet.eval.label_audit import corrected_boxes, flagged_annotation_ids
from platedet.eval.metrics import READABLE_PLATE_WIDTH
from platedet.postprocess import DEFAULT_TILE_OVERLAP, Detection

LOGGER = logging.getLogger("platedet.eval")
DEFAULT_DATASET_DIR = Path("data/processed/combined")
DEFAULT_HOLDOUT_DIR = Path("data/processed/holdout")

# (label, split, class, metric path, comparison, target) from SPEC §6.
TARGETS = (
    ("Plate recall @0.2 on test", "test", "license_plate", ("recall",), ">=", 0.95),
    ("Plate recall @0.2 on holdout", "holdout", "license_plate", ("recall",), ">=", 0.95),
    (
        "Small-plate recall (12-32 px) on test",
        "test",
        "license_plate",
        ("small_plates", "recall"),
        ">=",
        0.85,
    ),
    ("Vehicle recall @0.3 on test", "test", "vehicle", ("recall",), ">=", 0.90),
    (
        "Plate false positives per image on test",
        "test",
        "license_plate",
        ("false_positives_per_image",),
        "<=",
        1.0,
    ),
)


def target_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, split, class_name, path, comparison, target in TARGETS:
        value: Any = (metrics.get(split) or {}).get(class_name)
        for key in path:
            value = None if value is None else value.get(key)
        met = (
            None if value is None else (value >= target if comparison == ">=" else value <= target)
        )
        rows.append(
            {"metric": label, "value": value, "target": f"{comparison} {target}", "met": met}
        )
    return rows


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def write_report(run_dir: Path, summary: dict[str, Any], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or run_dir / "eval"
    lines = [f"# Evaluation: {summary['run_id']}", ""]
    if summary.get("tiles"):
        lines += [f"Tiled detection: {summary['tiles']} (ADR-0013).", ""]
    lines += ["## Targets (SPEC §6)", ""]
    all_rows = summary.get("targets_all_plates")
    audited_rows = summary.get("targets_audited")
    header = "| Metric | Value | Target | Met |"
    if all_rows:
        lines += [
            f"Plates under {summary['min_plate_width']:g} px wide are ignored (ADR-0013). "
            "*All plates* counts every plate, as SPEC §6 is currently written.",
            "",
        ]
        header += " All plates |"
    if audited_rows:
        flagged = ", ".join(f"{s}: {n}" for s, n in summary["audit_flagged"].items())
        corrected = (
            ", ".join(f"{s}: {n}" for s, n in (summary.get("audit_corrected") or {}).items())
            or "none"
        )
        lines += [
            "*Audited* also ignores labels the plate label audit flagged as loose or "
            f"not a plate and uses redrawn boxes (ADR-0014; flagged {flagged}; "
            f"redrawn {corrected}).",
            "",
        ]
        header += " Audited | Met (audited) |"
    lines += [header, "|" + "---|" * (header.count("|") - 1)]
    for index, row in enumerate(summary["targets"]):
        met = {True: "yes", False: "**no**", None: "n/a"}[row["met"]]
        line = f"| {row['metric']} | {_fmt(row['value'])} | {row['target']} | {met} |"
        if all_rows:
            line += f" {_fmt(all_rows[index]['value'])} |"
        if audited_rows:
            audited = audited_rows[index]
            audited_met = {True: "yes", False: "**no**", None: "n/a"}[audited["met"]]
            line += f" {_fmt(audited['value'])} | {audited_met} |"
        lines.append(line)
    lines += [
        "",
        "## All metrics",
        "",
        "| Split | Class | Recall | GT | Missed | FP/img | AP@0.5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for split, split_metrics in summary["metrics"].items():
        for class_name in ("license_plate", "vehicle"):
            entry = split_metrics.get(class_name)
            if not entry or entry["recall"] is None:
                continue
            fp = entry["false_positives_per_image"]
            lines.append(
                f"| {split} | {class_name} | {entry['recall']:.3f} | {entry['ground_truth']} | "
                f"{entry['missed']} | {fp:.2f} | {entry['ap50']:.3f} |"
            )
            small = entry.get("small_plates")
            if small and small["recall"] is not None:
                lines.append(
                    f"| {split} | small plate | {small['recall']:.3f} | "
                    f"{small['ground_truth']} | {small['missed']} | | |"
                )
    lines += ["", "## Worst misses (largest boxes first)", ""]
    for split, by_class in summary["worst_misses"].items():
        for class_name, misses in by_class.items():
            if not misses:
                continue
            lines.append(f"### {split} / {class_name} (gallery: `misses_{split}_{class_name}.jpg`)")
            lines.append("")
            for miss in misses[:10]:
                lines.append(
                    f"- {miss['rank']}. `{miss['file_name']}` box {miss['box_xyxy']} "
                    f"({miss['width_px']} px wide)"
                )
            lines.append("")
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class SavedPredictor:
    """Detections from an earlier run's `predictions_<split>.jsonl`, looked up by image path."""

    def __init__(self, out_dir: Path, split_dirs: dict[str, Path]) -> None:
        self.detections: dict[str, list[Detection]] = {}
        for split, split_dir in split_dirs.items():
            path = out_dir / f"predictions_{split}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"--rescore needs {path}")
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                key = str((split_dir / record["file_name"]).resolve())
                self.detections[key] = [
                    Detection(*map(float, d[:5]), int(d[5])) for d in record["detections"]
                ]

    def __call__(self, image: Any) -> list[Detection]:
        return self.detections[str(Path(image.filename).resolve())]


def evaluate_run(
    run_dir: Path,
    *,
    splits: tuple[str, ...] = ("val", "test", "holdout"),
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    holdout_dir: Path = DEFAULT_HOLDOUT_DIR,
    predictor: Any = None,
    tiles: int = 1,
    overlap: float = DEFAULT_TILE_OVERLAP,
    rescore: bool = False,
) -> dict[str, Any]:
    run_config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    model_path = run_dir / "model.onnx"
    config = {
        "name": run_dir.name,
        "model": {"path": str(model_path), "class_map": None},
        "dataset_dir": str(dataset_dir),
        "holdout_dir": str(holdout_dir),
        "splits": list(splits),
        "plate_conf": 0.2,
        "vehicle_conf": 0.3,
        "gallery_size": 30,
        "min_plate_width": READABLE_PLATE_WIDTH,
    }
    nms_iou = float(run_config.get("nms_iou", 0.45))
    split_dirs = {s: holdout_dir if s == "holdout" else dataset_dir for s in splits}
    audit = {
        split: ids
        for split in splits
        if (ids := flagged_annotation_ids(split, split_dirs[split])) is not None
    }
    if audit:
        config["audit_ignored_ids"] = audit
        config["audit_corrections"] = {split: corrected_boxes(split) for split in audit}
    out_dir = run_dir / ("eval" if tiles == 1 else f"eval_tiles{tiles}")
    if rescore:
        predictor = SavedPredictor(out_dir, split_dirs)
    if predictor is None:
        predictor = OnnxYoloxPredictor(
            model_path,
            input_size=int(run_config["input_size"]),
            channel_order=str(run_config.get("channel_order", "bgr")),
            conf_threshold=0.01,
            nms_iou=nms_iou,
            class_map=None,
        )
    if tiles > 1 and not rescore:
        predictor = TiledPredictor(predictor, grid=tiles, overlap=overlap, nms_iou=nms_iou)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = evaluate(config, predictor, out_dir)
    summary = {
        "run_id": run_dir.name,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "git": _git_state(),
        "tiles": None if tiles == 1 else {"grid": tiles, "overlap": overlap},
        "min_plate_width": READABLE_PLATE_WIDTH,
        "audit_flagged": {split: len(ids) for split, ids in audit.items()},
        "audit_corrected": {s: len(b) for s, b in (config.get("audit_corrections") or {}).items()},
        "model_sha256": sha256_file(model_path) if model_path.is_file() else None,
        "manifest_sha256": {
            str(dataset_dir): sha256_file(dataset_dir / "manifest.csv"),
            **(
                {str(holdout_dir): sha256_file(holdout_dir / "manifest.csv")}
                if "holdout" in splits
                else {}
            ),
        },
        **results,
    }
    summary["targets"] = target_rows(summary["metrics"])
    if "metrics_all_plates" in summary:
        summary["targets_all_plates"] = target_rows(summary["metrics_all_plates"])
    if "metrics_audited" in summary:
        summary["targets_audited"] = target_rows(summary["metrics_audited"])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(run_dir, summary, out_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run ID under runs/")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--tiles", type=int, default=1, help="N for whole image + N x N tiles")
    parser.add_argument("--overlap", type=float, default=DEFAULT_TILE_OVERLAP)
    parser.add_argument("--splits", default="val,test,holdout")
    parser.add_argument("--rescore", action="store_true", help="reuse saved predictions")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = args.runs_dir / args.run
    evaluate_run(
        run_dir,
        splits=tuple(args.splits.split(",")),
        tiles=args.tiles,
        overlap=args.overlap,
        rescore=args.rescore,
    )
    print(run_dir / ("eval" if args.tiles == 1 else f"eval_tiles{args.tiles}") / "report.md")


if __name__ == "__main__":
    main()

"""Evaluate a trained run's ONNX model on val, test and holdout (`make eval RUN=<run_id>`).

Fixed SPEC §6 thresholds, nothing tuned. Writes `runs/<run_id>/eval/` with summary.json,
predictions, miss galleries and report.md (metrics against the v1.0 targets and the worst
misses with image paths).
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from platedet.eval.baseline import OnnxYoloxPredictor, _git_state, evaluate, sha256_file

LOGGER = logging.getLogger("platedet.eval")
DEFAULT_DATASET_DIR = Path("data/processed/combined")
DEFAULT_HOLDOUT_DIR = Path("data/processed/holdout")

# (label, split, class, metric path, comparison, target) from SPEC §6.
TARGETS = (
    ("Plate recall @0.2 on test", "test", "license_plate", ("recall",), ">=", 0.95),
    ("Plate recall @0.2 on holdout", "holdout", "license_plate", ("recall",), ">=", 0.95),
    (
        "Small-plate recall (<32 px) on test",
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


def write_report(run_dir: Path, summary: dict[str, Any]) -> Path:
    lines = [f"# Evaluation: {summary['run_id']}", "", "## Targets (SPEC §6)", ""]
    lines += ["| Metric | Value | Target | Met |", "|---|---|---|---|"]
    for row in summary["targets"]:
        value = "n/a" if row["value"] is None else f"{row['value']:.3f}"
        met = {True: "yes", False: "**no**", None: "n/a"}[row["met"]]
        lines.append(f"| {row['metric']} | {value} | {row['target']} | {met} |")
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
                    f"| {split} | plate < 32 px | {small['recall']:.3f} | "
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
    path = run_dir / "eval" / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def evaluate_run(
    run_dir: Path,
    *,
    splits: tuple[str, ...] = ("val", "test", "holdout"),
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    holdout_dir: Path = DEFAULT_HOLDOUT_DIR,
    predictor: Any = None,
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
    }
    if predictor is None:
        predictor = OnnxYoloxPredictor(
            model_path,
            input_size=int(run_config["input_size"]),
            channel_order=str(run_config.get("channel_order", "bgr")),
            conf_threshold=0.01,
            nms_iou=float(run_config.get("nms_iou", 0.45)),
            class_map=None,
        )
    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = evaluate(config, predictor, out_dir)
    summary = {
        "run_id": run_dir.name,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "git": _git_state(),
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
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(run_dir, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run ID under runs/")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate_run(args.runs_dir / args.run)
    print(args.runs_dir / args.run / "eval" / "report.md")


if __name__ == "__main__":
    main()

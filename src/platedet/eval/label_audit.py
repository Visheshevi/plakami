"""Plate label audit for the evaluation splits (ADR-0014).

`prepare` draws numbered review sheets: one zoomed crop per labelled plate (at least
`READABLE_PLATE_WIDTH` wide) with the label box in red. Sheets never show model output, so
decisions don't depend on whether the model found the plate.

Each box gets one decision:
  ok         the box fits a real plate,
  loose      a real plate, but the box is clearly too big, too small or shifted,
  not_plate  no readable plate in the box (empty holder, sign, blank or covered area),
  fixed      the box was redrawn on the owner's request (ADR-0014).

Redrawn boxes are labels, so they stay with the dataset in `data/` (`corrections.json`,
never committed); `make audit-corrections` checks them against the `fixed` decisions and
records the file's hash in the committed meta file.

`finalize` checks the filled sheet covers every box exactly once and writes the committed
decisions file (`configs/audit/plate_labels.csv`: box IDs, decisions and notes only, no
boxes or images) plus a meta file with the annotation hashes it was made against.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from platedet.eval.baseline import sha256_file
from platedet.eval.metrics import PLATE, READABLE_PLATE_WIDTH

LOGGER = logging.getLogger("platedet.eval")
DECISIONS = ("ok", "loose", "not_plate", "fixed")
FLAGGED = frozenset({"loose", "not_plate"})
SPLITS = {"test": Path("data/processed/combined"), "holdout": Path("data/processed/holdout")}
AUDIT_DIR = Path("data/processed/label_audit")
DECISIONS_PATH = Path("configs/audit/plate_labels.csv")
META_PATH = Path("configs/audit/plate_labels.meta.json")
CORRECTIONS_PATH = AUDIT_DIR / "corrections.json"
TEMPLATE_FIELDS = ("box_id", "sheet", "number", "width_px", "decision", "note")
TILE_W, TILE_H, COLUMNS, PER_SHEET = 360, 270, 4, 20


def box_id(split: str, annotation_id: int) -> str:
    return f"{split}/{annotation_id}"


def _annotation_path(split: str, dataset_dir: Path) -> Path:
    return dataset_dir / "annotations" / f"{split}.json"


def audit_boxes(split: str, dataset_dir: Path) -> list[dict[str, Any]]:
    """Readable plate labels in a split, in annotation order."""
    dataset = json.loads(_annotation_path(split, dataset_dir).read_text(encoding="utf-8"))
    images = {image["id"]: image for image in dataset["images"]}
    boxes = []
    for annotation in dataset["annotations"]:
        x, y, width, height = annotation["bbox"]
        if annotation["category_id"] != PLATE or width < READABLE_PLATE_WIDTH:
            continue
        boxes.append(
            {
                "box_id": box_id(split, annotation["id"]),
                "file_name": images[annotation["image_id"]]["file_name"],
                "xyxy": (x, y, x + width, y + height),
            }
        )
    return boxes


def crop_tile(image: Image.Image, xyxy: tuple[float, float, float, float]) -> Image.Image:
    """The box with one box-size of context on each side (at least 40 px), box in red."""
    x1, y1, x2, y2 = xyxy
    pad_x, pad_y = max(x2 - x1, 40.0), max(y2 - y1, 40.0)
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right = min(image.width, int(x2 + pad_x) + 1)
    bottom = min(image.height, int(y2 + pad_y) + 1)
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    scale = min((TILE_W - 4) / crop.width, (TILE_H - 22) / crop.height)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))))
    ImageDraw.Draw(crop).rectangle(
        [(x1 - left) * scale, (y1 - top) * scale, (x2 - left) * scale, (y2 - top) * scale],
        outline="red",
        width=2,
    )
    return crop


def prepare(out_dir: Path = AUDIT_DIR, splits: dict[str, Path] = SPLITS) -> dict[str, int]:
    """Write sheets `<split>_NNN.jpg` and `review_template.csv` under `out_dir`."""
    sheets_dir = out_dir / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for split, dataset_dir in splits.items():
        boxes = audit_boxes(split, dataset_dir)
        counts[split] = len(boxes)
        for sheet_number, start in enumerate(range(0, len(boxes), PER_SHEET), start=1):
            chunk = boxes[start : start + PER_SHEET]
            sheet_rows = (len(chunk) + COLUMNS - 1) // COLUMNS
            sheet = Image.new("RGB", (COLUMNS * TILE_W, sheet_rows * TILE_H), "white")
            draw = ImageDraw.Draw(sheet)
            name = f"{split}_{sheet_number:03d}"
            for index, box in enumerate(chunk):
                with Image.open(dataset_dir / box["file_name"]) as source:
                    tile = crop_tile(source, box["xyxy"])
                x, y = (index % COLUMNS) * TILE_W, (index // COLUMNS) * TILE_H
                sheet.paste(tile, (x + 2, y + 2))
                draw.text((x + 4, y + TILE_H - 18), str(index + 1), fill="black")
                rows.append(
                    {
                        "box_id": box["box_id"],
                        "sheet": name,
                        "number": index + 1,
                        "width_px": round(box["xyxy"][2] - box["xyxy"][0], 1),
                        "decision": "",
                        "note": "",
                    }
                )
            sheet.save(sheets_dir / f"{name}.jpg", quality=90)
    with (out_dir / "review_template.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TEMPLATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Audit sheets in %s: %s", sheets_dir, counts)
    return counts


def read_decisions(path: Path) -> dict[str, tuple[str, str]]:
    """`box_id -> (decision, note)`; every decision must be one of DECISIONS."""
    decisions: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["box_id"].strip()
            decision = row["decision"].strip().lower()
            if decision not in DECISIONS:
                raise ValueError(f"Invalid audit decision for {key}: {decision!r}")
            if key in decisions:
                raise ValueError(f"Duplicate audit row for {key}")
            decisions[key] = (decision, (row.get("note") or "").strip())
    return decisions


def _hashes(splits: dict[str, Path]) -> dict[str, str]:
    return {
        split: sha256_file(_annotation_path(split, dataset_dir))
        for split, dataset_dir in splits.items()
    }


def finalize(
    review_path: Path,
    decisions_path: Path = DECISIONS_PATH,
    meta_path: Path = META_PATH,
    splits: dict[str, Path] = SPLITS,
) -> Counter[str]:
    """Check the review covers every audited box once; write the committed decisions."""
    decisions = read_decisions(review_path)
    expected = {box["box_id"] for split, d in splits.items() for box in audit_boxes(split, d)}
    missing, extra = expected - decisions.keys(), decisions.keys() - expected
    if missing or extra:
        raise ValueError(
            f"Audit covers the wrong boxes: {len(missing)} missing, {len(extra)} extra"
        )
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    with decisions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("box_id", "decision", "note"))
        for key in sorted(decisions, key=lambda k: (k.split("/")[0], int(k.split("/")[1]))):
            writer.writerow((key, *decisions[key]))
    meta = {"annotations_sha256": _hashes(splits), "readable_plate_width": READABLE_PLATE_WIDTH}
    if meta_path.is_file():
        # Keep the hash of earlier box corrections; `record_corrections` re-checks them.
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if "corrections_sha256" in previous:
            meta["corrections_sha256"] = previous["corrections_sha256"]
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    counts = Counter(decision for decision, _ in decisions.values())
    LOGGER.info("Audit decisions written to %s: %s", decisions_path, dict(counts))
    return counts


def flagged_annotation_ids(
    split: str,
    dataset_dir: Path,
    decisions_path: Path = DECISIONS_PATH,
    meta_path: Path = META_PATH,
) -> frozenset[int] | None:
    """Annotation IDs marked loose or not_plate in `split`, or None if there is no audit.

    Refuses decisions made against different annotations (a rebuilt dataset).
    """
    if not decisions_path.is_file() or not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = meta["annotations_sha256"].get(split)
    if expected is None:
        return None
    if sha256_file(_annotation_path(split, dataset_dir)) != expected:
        raise ValueError(f"Label audit was made for different {split} annotations; redo it")
    return frozenset(
        int(key.split("/")[1])
        for key, (decision, _) in read_decisions(decisions_path).items()
        if key.split("/")[0] == split and decision in FLAGGED
    )


def _read_corrections(path: Path) -> dict[str, list[float]]:
    boxes = json.loads(path.read_text(encoding="utf-8"))["boxes"]
    for key, box in boxes.items():
        if len(box) != 4 or not (box[0] < box[2] and box[1] < box[3]):
            raise ValueError(f"Correction for {key} is not an x1, y1, x2, y2 box: {box}")
    return boxes


def _fixed_ids(decisions_path: Path) -> set[str]:
    return {k for k, (d, _) in read_decisions(decisions_path).items() if d == "fixed"}


def record_corrections(
    corrections_path: Path = CORRECTIONS_PATH,
    decisions_path: Path = DECISIONS_PATH,
    meta_path: Path = META_PATH,
) -> int:
    """Check corrections match the `fixed` decisions one to one; store their hash in meta."""
    boxes = _read_corrections(corrections_path)
    fixed = _fixed_ids(decisions_path)
    if set(boxes) != fixed:
        raise ValueError(f"Corrections don't match 'fixed' decisions: {sorted(set(boxes) ^ fixed)}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["corrections_sha256"] = sha256_file(corrections_path)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return len(boxes)


def corrected_boxes(
    split: str,
    decisions_path: Path = DECISIONS_PATH,
    meta_path: Path = META_PATH,
    corrections_path: Path = CORRECTIONS_PATH,
) -> dict[int, tuple[float, float, float, float]]:
    """Redrawn boxes for `split` by annotation ID (empty if nothing was fixed).

    Refuses a corrections file that differs from the one recorded in the meta file.
    """
    if not decisions_path.is_file() or not _fixed_ids(decisions_path):
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not corrections_path.is_file():
        raise FileNotFoundError(f"Label audit has fixed boxes but {corrections_path} is missing")
    if sha256_file(corrections_path) != meta.get("corrections_sha256"):
        raise ValueError(
            f"{corrections_path} differs from the recorded one; rerun audit-corrections"
        )
    return {
        int(key.split("/")[1]): tuple(float(v) for v in box)  # type: ignore[misc]
        for key, box in _read_corrections(corrections_path).items()
        if key.split("/")[0] == split
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    final = sub.add_parser("finalize")
    final.add_argument("--review", type=Path, default=AUDIT_DIR / "review.csv")
    sub.add_parser("corrections")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "prepare":
        prepare()
    elif args.command == "finalize":
        finalize(args.review)
    else:
        LOGGER.info("Recorded %d box corrections", record_corrections())


if __name__ == "__main__":
    main()

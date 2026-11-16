"""Release package helpers (ADR-0018) and the integration steps documented in README.md."""

import math
import re
from pathlib import Path

import numpy as np
import pytest

from platedet.export.release import (
    _check_same,
    boxes_match,
    render_model_card,
    training_attribution,
    verify_sha256sums,
    write_sha256sums,
)
from platedet.postprocess import postprocess

ROOT = Path(__file__).resolve().parents[1]


def test_sha256sums_round_trip_and_tampering(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"\x00\x01")
    write_sha256sums(tmp_path)
    lines = (tmp_path / "SHA256SUMS").read_text().splitlines()
    assert [line.split("  ")[1] for line in lines] == ["a.txt", "sub/b.bin"]
    assert verify_sha256sums(tmp_path) == []

    (tmp_path / "a.txt").write_text("changed")
    (tmp_path / "extra.txt").write_text("x")
    (tmp_path / "sub" / "b.bin").unlink()
    assert sorted(verify_sha256sums(tmp_path)) == [
        "checksum differs: a.txt",
        "missing: sub/b.bin",
        "not in SHA256SUMS: extra.txt",
    ]


def test_check_same_refuses_a_different_model_file(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"model")
    with pytest.raises(ValueError, match="not the file evaluated"):
        _check_same(path, "0" * 64, "evaluated")


def test_boxes_match_uses_tolerances_and_order():
    expected = [{"class": "license_plate", "score": 0.8, "box": [10, 10, 50, 20]}]
    close = [{"class": "license_plate", "score": 0.79, "box": [11, 10, 50, 21]}]
    assert boxes_match(expected, close, px=2, score=0.05)
    assert not boxes_match(expected, close, px=0.5, score=0.001)
    assert not boxes_match(expected, [], px=2, score=0.05)
    other = [{**close[0], "class": "vehicle"}]
    assert not boxes_match(expected, other, px=2, score=0.05)


def _manifest():
    split = {
        "images": 1191,
        "plates": 1344,
        "plate_recall": 0.95,
        "small_plate_recall": 0.87,
        "plate_false_positives_per_image": 0.5,
        "vehicle_recall": None,
        "vehicle_false_positives_per_image": None,
    }
    parity_mode = {
        c: {"reference_boxes": 10, "off_tolerance": 0, "missing": 0}
        for c in ("license_plate", "vehicle")
    }
    return {
        "version": "1.0.0",
        "labels": ["license_plate", "vehicle"],
        "recommended_thresholds": {"license_plate": 0.1, "vehicle": 0.3},
        "files": {"detector.onnx": {"bytes": 20e6}, "detector_fp16.tflite": {"bytes": 10e6}},
        "metrics": {
            "at_recommended_thresholds": {"test": split, "holdout": split},
            "spec_targets": [{"metric": "m", "value": 0.9, "target": ">= 0.95", "met": False}],
        },
        "parity": {
            "images": 1435,
            "fp16": {"single": parity_mode, "tiles2": parity_mode},
            "speed_ms_mac_cpu": {
                k: {"median_ms_per_pass": 50, "median_ms_per_photo_2x2_tiles": 400}
                for k in ("onnx", "fp16")
            },
        },
        "provenance": {"checkpoint_sha256": "abc", "release_commit": {"commit": "def"}},
    }


def test_model_card_template_fills_every_placeholder():
    template = (ROOT / "docs" / "release" / "MODEL_CARD.md.tmpl").read_text()
    card = render_model_card(template, _manifest())
    assert "$" not in card
    assert "| m | 0.900 | >= 0.95 | no |" in card
    assert "1,191 photos" in card and "| Vehicle recall | — | — |" in card


def test_training_attribution_replaces_the_heading(tmp_path):
    source = tmp_path / "ATTRIBUTION.md"
    source.write_text("# Dataset attribution\n\n- one: [a](u) — [CC BY 2.0](l)\n")
    text = training_attribution(source)
    assert text.startswith("# Attribution\n") and "Dataset attribution" not in text
    assert text.rstrip().endswith("- one: [a](u) — [CC BY 2.0](l)")


def _readme_decode(output, ratio, width, height, thresholds=(0.10, 0.30), size=640):
    """Steps 5-6 exactly as README.md describes them, without postprocess.py."""
    found, row = [], 0
    for stride in (8, 16, 32):
        cells = size // stride
        for gy in range(cells):
            for gx in range(cells):
                r = output[row]
                row += 1
                cls = 0 if r[5] >= r[6] else 1
                score = float(r[4] * r[5 + cls])
                if score < thresholds[cls]:
                    continue
                cx, cy = (r[0] + gx) * stride, (r[1] + gy) * stride
                w, h = math.exp(r[2]) * stride, math.exp(r[3]) * stride
                box = [(cx - w / 2) / ratio, (cy - h / 2) / ratio]
                box += [(cx + w / 2) / ratio, (cy + h / 2) / ratio]
                box = [min(max(box[0], 0), width), min(max(box[1], 0), height)] + [
                    min(max(box[2], 0), width),
                    min(max(box[3], 0), height),
                ]
                found.append((*box, score, cls))
    kept = []
    for cls in sorted({d[5] for d in found}):
        order = sorted((d for d in found if d[5] == cls), key=lambda d: -d[4])
        while order:
            best = order.pop(0)
            kept.append(best)
            order = [d for d in order if _iou(best, d) <= 0.45]
    return sorted(kept, key=lambda d: -d[4])


def _iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / union if union > 0 else 0.0


def test_readme_decoding_matches_the_reference_postprocess():
    rng = np.random.default_rng(0)
    raw = np.zeros((1, 8400, 7), dtype=np.float32)
    raw[0, :, :2] = rng.uniform(-0.5, 1.5, (8400, 2))
    raw[0, :, 2:4] = rng.uniform(-1, 2, (8400, 2))
    raw[0, :, 4:] = rng.uniform(0, 1, (8400, 3)) ** 3  # mostly low scores, some high
    ratio, width, height = 0.4, 1600, 1200
    reference = postprocess(
        raw, size=640, ratio=ratio, image_width=width, image_height=height, conf_threshold=0.01
    )
    thresholds = {1: 0.10, 2: 0.30}
    reference = [d for d in reference if d.score >= thresholds[d.class_id]]
    readme = _readme_decode(raw[0], ratio, width, height)
    assert len(reference) == len(readme) > 20
    for ref, ours in zip(reference, readme, strict=True):
        assert ref.class_id == ours[5] + 1
        assert ref.score == pytest.approx(ours[4], abs=1e-6)
        assert np.allclose([ref.x1, ref.y1, ref.x2, ref.y2], ours[:4], atol=1e-3)


def test_readme_numbers_match_the_release_config():
    import yaml

    readme = (ROOT / "README.md").read_text()
    release = yaml.safe_load((ROOT / "configs" / "release" / "v1.yaml").read_text())
    thresholds = release["recommended_thresholds"]
    assert f"plate {thresholds['license_plate']:.2f}" in readme
    assert f"vehicle {thresholds['vehicle']:.2f}" in readme
    assert f"**{release['nms_iou']}**" in readme
    overlap = release["tiling"]["overlap"]
    grid = release["tiling"]["grid"]
    assert f"dim / {grid - (grid - 1) * overlap:g}" in readme
    assert re.search(r"floatArrayOf\(0\.10f, 0\.30f\)", readme)

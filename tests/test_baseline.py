import json

import pytest
import yaml
from PIL import Image

from platedet.eval.baseline import _prepare_model, run_baseline
from platedet.postprocess import Detection


def _dataset(tmp_path):
    dataset_dir = tmp_path / "combined"
    (dataset_dir / "images" / "src").mkdir(parents=True)
    (dataset_dir / "annotations").mkdir()
    Image.new("RGB", (100, 80), "gray").save(dataset_dir / "images/src/a.jpg")
    Image.new("RGB", (100, 80), "gray").save(dataset_dir / "images/src/b.jpg")
    coco = {
        "images": [
            {
                "id": 1,
                "file_name": "images/src/a.jpg",
                "width": 100,
                "height": 80,
                "annotated_classes": ["license_plate", "vehicle"],
            },
            {
                "id": 2,
                "file_name": "images/src/b.jpg",
                "width": 100,
                "height": 80,
                "annotated_classes": ["license_plate"],
            },
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [10, 10, 60, 50]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [30, 40, 20, 8]},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [5, 5, 20, 8]},
        ],
        "categories": [{"id": 1, "name": "license_plate"}, {"id": 2, "name": "vehicle"}],
    }
    (dataset_dir / "annotations" / "val.json").write_text(json.dumps(coco))
    (dataset_dir / "manifest.csv").write_text("image_id\nx\n")
    config = {
        "name": "fake",
        "model": {"sha256": "abc", "class_map": {2: 2}},
        "dataset_dir": str(dataset_dir),
        "splits": ["val"],
        "plate_conf": 0.2,
        "vehicle_conf": 0.3,
        "gallery_size": 5,
    }
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_baseline_writes_summary_with_vehicle_metrics_and_no_plate_metrics(tmp_path):
    config_path = _dataset(tmp_path)

    def predictor(image):
        return [Detection(10, 10, 70, 60, 0.9, 2), Detection(0, 0, 5, 5, 0.9, 2)]

    run_dir = run_baseline(config_path, runs_dir=tmp_path / "runs", predictor=predictor)
    summary = json.loads((run_dir / "summary.json").read_text())
    vehicle = summary["metrics"]["val"]["vehicle"]
    assert vehicle["recall"] == 1.0 and vehicle["images"] == 1
    # The stray box on the plate-only image is not a false positive.
    assert vehicle["false_positives_per_image"] == 1.0
    assert summary["metrics"]["val"]["license_plate"] is None
    assert summary["checkpoint_sha256"] == "abc" and len(summary["dataset_manifest_sha256"]) == 1
    assert (run_dir / "config.yaml").is_file()
    assert len((run_dir / "predictions_val.jsonl").read_text().splitlines()) == 2


def test_baseline_gallery_lists_misses(tmp_path):
    config_path = _dataset(tmp_path)
    run_dir = run_baseline(config_path, runs_dir=tmp_path / "runs", predictor=lambda image: [])
    summary = json.loads((run_dir / "summary.json").read_text())
    (miss,) = summary["worst_misses"]["val"]["vehicle"]
    assert miss["file_name"] == "images/src/a.jpg" and miss["box_xyxy"] == [10, 10, 70, 60]
    assert (run_dir / "misses_val_vehicle.jpg").is_file()


def test_prepare_model_rejects_wrong_checkpoint_hash(tmp_path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"not the real model")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _prepare_model({"path": str(model_path), "url": "https://example.invalid", "sha256": "0"})


def test_baseline_scores_holdout_from_its_own_folder(tmp_path):
    config_path = _dataset(tmp_path)
    holdout = tmp_path / "holdout"
    (holdout / "annotations").mkdir(parents=True)
    (holdout / "images").mkdir()
    Image.new("RGB", (50, 50)).save(holdout / "images" / "h.jpg")
    (holdout / "manifest.csv").write_text("image_id\nh\n")
    (holdout / "annotations" / "holdout.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "images/h.jpg",
                        "width": 50,
                        "height": 50,
                        "annotated_classes": ["license_plate"],
                    }
                ],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 20, 8]}],
                "categories": [{"id": 1, "name": "license_plate"}, {"id": 2, "name": "vehicle"}],
            }
        )
    )
    config = yaml.safe_load(config_path.read_text())
    config.update(splits=["val", "holdout"], holdout_dir=str(holdout))
    config_path.write_text(yaml.safe_dump(config))
    run_dir = run_baseline(config_path, runs_dir=tmp_path / "runs", predictor=lambda image: [])
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["metrics"]["holdout"]["images"] == 1
    assert summary["metrics"]["holdout"]["vehicle"]["images"] == 0
    assert len(summary["dataset_manifest_sha256"]) == 2

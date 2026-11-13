import json

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

from platedet.eval.evaluate_run import target_rows, write_report
from platedet.postprocess import Detection
from platedet.train.dataset import build_training_data, convert_split
from platedet.train.pseudo_labels import pseudo_label_split

CATEGORIES = [{"id": 1, "name": "license_plate"}, {"id": 2, "name": "vehicle"}]


def _coco():
    return {
        "images": [
            {
                "id": 1,
                "file_name": "images/oi/a.jpg",
                "width": 1000,
                "height": 500,
                "annotated_classes": ["license_plate", "vehicle"],
            },
            {
                "id": 2,
                "file_name": "images/ccpd/b.jpg",
                "width": 800,
                "height": 400,
                "annotated_classes": ["license_plate"],
            },
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 60, 20]},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [500, 300, 30, 8]},  # 8 * 0.64 < 6
            {"id": 3, "image_id": 1, "category_id": 2, "bbox": [0, 0, 400, 300]},
            {"id": 4, "image_id": 2, "category_id": 1, "bbox": [100, 100, 80, 25]},
        ],
        "categories": CATEGORIES,
    }


def test_convert_split_drops_small_boxes_and_adds_pseudo_labels_to_plate_only_images():
    pseudo = {"images/ccpd/b.jpg": [[50, 50, 450, 350, 0.9], [0, 0, 6, 6, 0.8]]}
    output, counts = convert_split(_coco(), pseudo, input_size=640, min_box_px=6)
    assert counts["dropped_small_license_plate"] == 1
    assert counts["kept_license_plate"] == 2
    assert counts["pseudo_vehicle_boxes"] == 1 and counts["dropped_small_vehicle"] == 1
    assert counts["pseudo_labelled_images"] == 1
    pseudo_boxes = [a for a in output["annotations"] if a.get("pseudo")]
    assert pseudo_boxes == [
        {
            "id": 4,
            "image_id": 2,
            "category_id": 2,
            "bbox": [50, 50, 400, 300],
            "area": 120000,
            "iscrowd": 0,
            "pseudo": True,
        }
    ]
    assert "annotated_classes" not in output["images"][0]


def test_convert_split_requires_pseudo_labels_for_plate_only_images():
    with pytest.raises(ValueError, match="No pseudo-labels"):
        convert_split(_coco(), {}, input_size=640, min_box_px=6)


def test_build_training_data_refuses_stale_pseudo_labels(tmp_path):
    combined = tmp_path / "combined"
    (combined / "annotations").mkdir(parents=True)
    (combined / "manifest.csv").write_text("image_id\n")
    pseudo = tmp_path / "pseudo.json"
    pseudo.write_text(json.dumps({"dataset_manifest_sha256": "old", "labels": {}}))
    config = {
        "input_size": 640,
        "data": {
            "combined_dir": str(combined),
            "pseudo_labels": str(pseudo),
            "training_dir": str(tmp_path / "out"),
        },
    }
    with pytest.raises(ValueError, match="different dataset"):
        build_training_data(config)


def test_pseudo_labels_only_for_images_without_vehicle_labels(tmp_path):
    (tmp_path / "annotations").mkdir()
    (tmp_path / "images" / "ccpd").mkdir(parents=True)
    (tmp_path / "images" / "oi").mkdir(parents=True)
    for name in ("ccpd/b.jpg", "oi/a.jpg"):
        Image.new("RGB", (40, 20)).save(tmp_path / "images" / name)
    (tmp_path / "annotations" / "train.json").write_text(json.dumps(_coco()))
    calls = []

    def predictor(image):
        calls.append(image.size)
        return [
            Detection(1, 1, 30, 15, 0.9, 2),
            Detection(0, 0, 5, 5, 0.3, 2),
            Detection(2, 2, 9, 9, 0.95, 1),
        ]

    labels = pseudo_label_split(tmp_path, "train", predictor, score_threshold=0.5)
    assert list(labels) == ["images/ccpd/b.jpg"] and len(calls) == 1
    assert labels["images/ccpd/b.jpg"] == [[1, 1, 30, 15, 0.9]]


def test_target_rows_and_report(tmp_path):
    metrics = {
        "test": {
            "license_plate": {
                "recall": 0.96,
                "ground_truth": 10,
                "missed": 1,
                "false_positives_per_image": 1.5,
                "ap50": 0.9,
                "small_plates": {"recall": 0.5, "ground_truth": 2, "missed": 1},
            },
            "vehicle": {
                "recall": 0.91,
                "ground_truth": 5,
                "missed": 0,
                "false_positives_per_image": 0.1,
                "ap50": 0.8,
            },
        }
    }
    rows = {row["metric"]: row for row in target_rows(metrics)}
    assert rows["Plate recall @0.2 on test"]["met"] is True
    assert rows["Plate recall @0.2 on holdout"]["met"] is None
    assert rows["Small-plate recall (<32 px) on test"]["met"] is False
    assert rows["Plate false positives per image on test"]["met"] is False
    (tmp_path / "eval").mkdir()
    summary = {
        "run_id": "r",
        "metrics": metrics,
        "targets": list(rows.values()),
        "worst_misses": {
            "test": {
                "license_plate": [
                    {
                        "rank": 1,
                        "file_name": "images/x.jpg",
                        "box_xyxy": [1, 2, 3, 4],
                        "width_px": 2.0,
                    }
                ]
            }
        },
    }
    report = write_report(tmp_path, summary).read_text()
    assert "| Plate recall @0.2 on test | 0.960 | >= 0.95 | yes |" in report
    assert "`images/x.jpg`" in report


@pytest.fixture
def synthetic_training_dir(tmp_path):
    """Six generated images with a 'vehicle' and a 'plate' each, in YOLOX training layout."""
    root = tmp_path / "yolox_64"
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    images, annotations = [], []
    rng = np.random.default_rng(0)
    for index in range(6):
        image = Image.new("RGB", (160, 120), tuple(int(v) for v in rng.integers(0, 255, 3)))
        draw = ImageDraw.Draw(image)
        draw.rectangle([20, 30, 140, 100], fill=(200, 30, 30))
        draw.rectangle([60, 80, 100, 92], fill=(250, 250, 250))
        name = f"images/{index}.jpg"
        image.save(root / name)
        images.append({"id": index + 1, "file_name": name, "width": 160, "height": 120})
        annotations += [
            {
                "id": 2 * index + 1,
                "image_id": index + 1,
                "category_id": 2,
                "bbox": [20, 30, 120, 70],
                "area": 8400,
                "iscrowd": 0,
            },
            {
                "id": 2 * index + 2,
                "image_id": index + 1,
                "category_id": 1,
                "bbox": [60, 80, 40, 12],
                "area": 480,
                "iscrowd": 0,
            },
        ]
    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    for split in ("train", "val"):
        (root / "annotations" / f"{split}.json").write_text(json.dumps(coco))
    return root


def test_smoke_train_step_and_onnx_export(synthetic_training_dir, tmp_path):
    torch = pytest.importorskip("torch")
    from platedet.export.onnx_export import export_onnx
    from platedet.postprocess import letterbox, postprocess
    from platedet.train.exp import make_exp

    config = {
        "name": "smoke",
        "seed": 0,
        "input_size": 64,
        "channel_order": "bgr",
        "data": {"training_dir": str(synthetic_training_dir)},
        "train": {"max_epoch": 1, "data_num_workers": 0},
    }
    run_dir = tmp_path / "run"
    (run_dir / "yolox").mkdir(parents=True)
    torch.manual_seed(0)
    exp = make_exp(config, run_dir)
    assert exp.num_classes == 2 and exp.input_size == (64, 64) and exp.max_epoch == 1

    dataset = exp.get_dataset()
    assert len(dataset) == 6
    samples = [dataset[index] for index in range(2)]
    images = torch.from_numpy(np.stack([sample[0] for sample in samples]))
    targets = torch.from_numpy(np.stack([sample[1] for sample in samples])).float()
    assert (targets.sum(dim=2) > 0).sum(dim=1).tolist() == [2, 2]  # both boxes survive

    model = exp.get_model()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    losses = model(images, targets)
    assert torch.isfinite(losses["total_loss"])
    losses["total_loss"].backward()
    optimizer.step()

    torch.save({"model": model.state_dict()}, run_dir / "yolox" / "best_ckpt.pth")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    onnx_path = export_onnx(run_dir)
    export_info = json.loads((run_dir / "export.json").read_text())
    assert export_info["max_abs_error_vs_torch"] <= 1e-3 and export_info["opset"] == 17

    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    image = Image.open(synthetic_training_dir / "images" / "0.jpg")
    tensor, ratio = letterbox(image, 64, "bgr")
    raw = session.run(None, {"images": tensor})[0]
    assert raw.shape == (1, 84, 7)
    detections = postprocess(
        raw, size=64, ratio=ratio, image_width=160, image_height=120, conf_threshold=0.0
    )
    assert all(d.class_id in (1, 2) for d in detections)


def _local_run_config(tmp_path, training_dir, max_epoch):
    torch = pytest.importorskip("torch")
    from platedet.eval.baseline import sha256_file
    from platedet.train.exp import make_exp

    (training_dir / "stats.json").write_text(
        json.dumps({"input_size": 64, "combined_manifest_sha256": "m", "pseudo_labels_sha256": "p"})
    )
    pretrained = tmp_path / "start.pth"
    if not pretrained.exists():
        config = {"name": "x", "input_size": 64, "data": {"training_dir": str(training_dir)}}
        torch.manual_seed(1)
        torch.save({"model": make_exp(config, tmp_path).get_model().state_dict()}, pretrained)
    config = {
        "name": "local_smoke",
        "seed": 0,
        "input_size": 64,
        "channel_order": "bgr",
        "pretrained": {"url": "unused", "path": str(pretrained), "sha256": sha256_file(pretrained)},
        "data": {"training_dir": str(training_dir)},
        "train": {
            "max_epoch": max_epoch,
            "no_aug_epochs": 1,
            "warmup_epochs": 1,
            "batch_size": 2,
            "fp16": False,
            "data_num_workers": 0,
            "eval_interval": 1,
            "print_interval": 1,
            "random_size": [2, 3],
        },
    }
    path = tmp_path / f"config_{max_epoch}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_local_trainer_trains_evaluates_and_resumes_on_cpu(synthetic_training_dir, tmp_path):
    torch = pytest.importorskip("torch")
    from platedet.train.train import train

    runs = tmp_path / "runs"
    first = _local_run_config(tmp_path, synthetic_training_dir, max_epoch=1)
    run_dir = train(first, runs_dir=runs, run_id="r", device="cpu")
    latest = torch.load(run_dir / "yolox" / "latest_ckpt.pth", weights_only=False)
    assert latest["start_epoch"] == 1
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["environment"]["device"] == "cpu"
    assert info["best_checkpoint_sha256"]

    second = _local_run_config(tmp_path, synthetic_training_dir, max_epoch=2)
    train(second, runs_dir=runs, run_id="r", device="cpu")
    latest = torch.load(run_dir / "yolox" / "latest_ckpt.pth", weights_only=False)
    assert latest["start_epoch"] == 2
    info = json.loads((run_dir / "run_info.json").read_text())
    assert len(info["resumed_at"]) == 1


def test_mps_type_names_only_patched_inside_context():
    torch = pytest.importorskip("torch")
    from platedet.train.local_trainer import mps_type_names

    original = torch.Tensor.type
    with mps_type_names():
        assert torch.Tensor.type is not original
        assert torch.zeros(1).type("torch.FloatTensor").dtype == torch.float32
    assert torch.Tensor.type is original


def test_local_trainer_runs_on_apple_gpu(synthetic_training_dir, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("no Apple GPU")
    from platedet.train.train import train

    config = _local_run_config(tmp_path, synthetic_training_dir, max_epoch=1)
    run_dir = train(config, runs_dir=tmp_path / "runs", run_id="mps", device="mps")
    info = json.loads((run_dir / "run_info.json").read_text())
    assert info["environment"]["device"] == "mps" and info["best_checkpoint_sha256"]

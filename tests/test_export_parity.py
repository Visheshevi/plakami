"""TFLite export helpers and the ONNX vs TFLite parity rules (ADR-0016)."""

import json

import numpy as np
from PIL import Image

from platedet.export.tflite_export import (
    calibration_images,
    onnx2tf_command,
    to_nhwc,
    write_calibration,
)
from platedet.parity.parity import THRESHOLDS, Tally, compare, passed
from platedet.postprocess import Detection, letterbox


def _dataset(tmp_path, names=("a", "b", "c", "d")):
    root = tmp_path / "combined"
    (root / "annotations").mkdir(parents=True)
    (root / "images" / "src").mkdir(parents=True)
    images = []
    rows = ["image_id,source,source_image_id,url,author,licence,licence_url,split"]
    for index, name in enumerate(names, start=1):
        Image.new("RGB", (40, 20), (index * 40, 0, 0)).save(root / "images" / "src" / f"{name}.jpg")
        images.append(
            {"id": index, "file_name": f"images/src/{name}.jpg", "width": 40, "height": 20}
        )
        rows.append(
            f"src:{name},src,src/{name}.jpg,http://x/{name},someone,CC BY 2.0,http://l,train"
        )
    (root / "annotations" / "train.json").write_text(json.dumps({"images": images}))
    (root / "manifest.csv").write_text("\n".join(rows) + "\n")
    return root


def test_calibration_images_are_a_stable_hash_ordered_train_subset(tmp_path):
    root = _dataset(tmp_path)
    first = calibration_images(root, 2)
    assert first == calibration_images(root, 2)
    assert len(first) == 2 and set(first) <= {f"images/src/{n}.jpg" for n in "abcd"}
    assert calibration_images(root, 4)[:2] == first  # growing the count keeps the prefix


def test_write_calibration_stores_nhwc_inputs_and_licences(tmp_path):
    root = _dataset(tmp_path)
    run_dir = tmp_path / "run"
    array_path, list_path = write_calibration(run_dir, root, 32, "bgr", 3)
    batch = np.load(array_path)
    assert batch.shape == (3, 32, 32, 3) and batch.dtype == np.float32
    lines = list_path.read_text().splitlines()
    assert lines[0] == "file_name,image_id,source,author,licence,licence_url,url"
    assert len(lines) == 4 and all("CC BY 2.0" in line for line in lines[1:])


def test_to_nhwc_matches_the_onnx_tensor():
    image = Image.new("RGB", (50, 30), (10, 20, 30))
    nchw, _ = letterbox(image, 64, "bgr")
    nhwc = to_nhwc(image, 64, "bgr")
    assert nhwc.shape == (1, 64, 64, 3)
    assert np.array_equal(nhwc[0, :, :, 0], nchw[0, 0])
    assert nhwc[0, 0, 0].tolist() == [30.0, 20.0, 10.0]  # BGR, no normalisation


def test_onnx2tf_command_pins_the_tool_and_disables_normalisation(tmp_path):
    plain = onnx2tf_command(tmp_path / "m.onnx", tmp_path / "out", None)
    assert plain[:6] == [
        "uvx",
        "--python",
        "3.12",
        "--from",
        "onnx2tf[tensorflow]==2.6.9",
        "onnx2tf",
    ]
    assert "-oiqt" not in plain and "tf_converter" in plain
    quant = onnx2tf_command(tmp_path / "m.onnx", tmp_path / "out", tmp_path / "c.npy")
    assert quant[quant.index("-cind") + 1 :][:4] == [
        "images",
        str(tmp_path / "c.npy"),
        "[[[[0.0,0.0,0.0]]]]",
        "[[[[1.0,1.0,1.0]]]]",
    ]


def _tallies():
    return {c: Tally() for c in THRESHOLDS}


def test_compare_counts_within_off_and_missing_boxes():
    reference = [
        Detection(0, 0, 100, 40, 0.90, 1),  # matched exactly
        Detection(200, 0, 300, 40, 0.50, 1),  # found, but confidence drifts by 0.10
        Detection(400, 0, 500, 40, 0.25, 1),  # missing in candidate
        Detection(600, 0, 700, 40, 0.10, 1),  # below the plate threshold: not checked
        Detection(0, 100, 300, 300, 0.80, 2),  # vehicle, box shifted (IoU < 0.9)
    ]
    candidate = [
        Detection(0, 0, 100, 40, 0.91, 1),
        Detection(200, 0, 300, 40, 0.40, 1),
        Detection(0, 130, 300, 330, 0.80, 2),
    ]
    tallies = _tallies()
    compare(reference, candidate, tallies, "x.jpg")
    plates, vehicles = tallies[1].as_dict(), tallies[2].as_dict()
    assert (plates["reference_boxes"], plates["within_tolerance"]) == (3, 1)
    assert (plates["off_tolerance"], plates["missing"]) == (1, 1)
    assert vehicles["off_tolerance"] == 1 and vehicles["missing"] == 0
    assert not passed({"license_plate": plates, "vehicle": vehicles})


def test_identical_outputs_pass():
    detections = [Detection(0, 0, 100, 40, 0.9, 1), Detection(0, 50, 200, 200, 0.7, 2)]
    tallies = _tallies()
    compare(detections, detections, tallies)
    assert passed({"license_plate": tallies[1].as_dict(), "vehicle": tallies[2].as_dict()})

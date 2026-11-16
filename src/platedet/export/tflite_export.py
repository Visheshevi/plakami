"""Convert a run's ONNX model to TFLite FP16 and INT8 with onnx2tf (SPEC §7, ADR-0016).

onnx2tf pins exact versions of numpy, onnxruntime and pytest that clash with this project,
so it runs in its own pinned tool environment (`uvx`), not in the project's venv. The
TensorFlow converter backend is used because it keeps float32 inputs and outputs and the
ONNX output layout:

  input  `images`: (1, S, S, 3) float32, BGR, 0-255 (channels-last; same values as ONNX)
  output `output`: (1, N, 5 + C) float32, same as the ONNX model

INT8 is full-integer inside with float32 input and output ("integer_quant"), calibrated on
`CALIBRATION_COUNT` train images chosen by a salted hash of the file name. The images used
are listed (with their manifest rows: source, author, licence) in `tflite/calibration.csv`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from platedet.eval.baseline import sha256_file
from platedet.postprocess import letterbox

ONNX2TF_VERSION = "2.6.9"
ONNX2TF = f"onnx2tf[tensorflow]=={ONNX2TF_VERSION}"
TOOL_PYTHON = "3.12"
CALIBRATION_COUNT = 300
CALIBRATION_SALT = "int8-calibration"
DEFAULT_DATASET_DIR = Path("data/processed/combined")
OUTPUTS = {"fp16": "model_float16.tflite", "int8": "model_integer_quant.tflite"}
RELEASE_NAMES = {"fp16": "detector_fp16.tflite", "int8": "detector_int8.tflite"}


def calibration_images(dataset_dir: Path, count: int = CALIBRATION_COUNT) -> list[str]:
    """`count` train file names, ordered by sha256(salt:file_name); never val/test/holdout."""
    train = json.loads((dataset_dir / "annotations" / "train.json").read_text(encoding="utf-8"))
    names = [image["file_name"] for image in train["images"]]

    def key(name: str) -> str:
        return hashlib.sha256(f"{CALIBRATION_SALT}:{name}".encode()).hexdigest()

    return sorted(names, key=key)[:count]


def to_nhwc(image: Image.Image, size: int, channel_order: str) -> np.ndarray:
    """The ONNX input tensor for `image`, transposed to the TFLite (1, S, S, 3) layout."""
    tensor, _ = letterbox(image, size, channel_order)
    return np.ascontiguousarray(tensor.transpose(0, 2, 3, 1))


def write_calibration(
    run_dir: Path, dataset_dir: Path, size: int, channel_order: str, count: int
) -> tuple[Path, Path]:
    """Calibration array (count, S, S, 3) float32 and the list of images with licences."""
    out = run_dir / "tflite"
    out.mkdir(parents=True, exist_ok=True)
    names = calibration_images(dataset_dir, count)
    batch = np.empty((len(names), size, size, 3), dtype=np.float32)
    for index, name in enumerate(names):
        with Image.open(dataset_dir / name) as image:
            batch[index] = to_nhwc(image, size, channel_order)[0]
    array_path = out / "calibration.npy"
    np.save(array_path, batch)

    with (dataset_dir / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = {row["source_image_id"]: row for row in csv.DictReader(stream)}
    list_path = out / "calibration.csv"
    fields = ("file_name", "image_id", "source", "author", "licence", "licence_url", "url")
    with list_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in names:
            row = rows[name.removeprefix("images/")]
            if row["split"] != "train":
                raise ValueError(f"Calibration image {name} is not in train")
            writer.writerow({"file_name": name, **{k: row[k] for k in fields[1:]}})
    return array_path, list_path


def onnx2tf_command(onnx_path: Path, out_dir: Path, calibration: Path | None) -> list[str]:
    command = [
        "uvx",
        "--python",
        TOOL_PYTHON,
        "--from",
        ONNX2TF,
        "onnx2tf",
        "-i",
        str(onnx_path),
        "-o",
        str(out_dir),
        "-tb",
        "tf_converter",
        "-coion",
    ]
    if calibration is not None:
        # No normalisation: the model takes raw 0-255 values (mean 0, std 1).
        command += [
            "-oiqt",
            "-qt",
            "per-channel",
            "-cind",
            "images",
            str(calibration),
            "[[[[0.0,0.0,0.0]]]]",
            "[[[[1.0,1.0,1.0]]]]",
        ]
    return command


def export_tflite(
    run_dir: Path,
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    calibration_count: int = CALIBRATION_COUNT,
    int8: bool = True,
) -> dict[str, Any]:
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    size = int(config["input_size"])
    channel_order = str(config.get("channel_order", "bgr"))
    onnx_path = run_dir / "model.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(f"{onnx_path} is missing; run the ONNX export first")

    calibration = calibration_list = None
    if int8:
        calibration, calibration_list = write_calibration(
            run_dir, dataset_dir, size, channel_order, calibration_count
        )
    work = run_dir / "tflite" / "onnx2tf"
    if work.exists():
        shutil.rmtree(work)
    log_path = run_dir / "tflite" / "onnx2tf.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = onnx2tf_command(onnx_path, work, calibration)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)

    files: dict[str, Any] = {}
    for kind, produced in OUTPUTS.items():
        if kind == "int8" and not int8:
            continue
        target = run_dir / RELEASE_NAMES[kind]
        shutil.copyfile(work / produced, target)
        files[kind] = {
            "file": target.name,
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
    info = {
        "onnx2tf": ONNX2TF_VERSION,
        "backend": "tf_converter",
        "command": command,
        "input": {"name": "images", "shape": [1, size, size, 3], "dtype": "float32"},
        "output": {"name": "output", "layout": "same as ONNX (1, N, 5 + classes)"},
        "files": files,
        "onnx_sha256": sha256_file(onnx_path),
    }
    if int8 and calibration_list is not None:
        info["int8_calibration"] = {
            "images": calibration_count,
            "split": "train",
            "selection": f"sha256('{CALIBRATION_SALT}:' + file_name), first {calibration_count}",
            "list": str(calibration_list.relative_to(run_dir)),
            "list_sha256": sha256_file(calibration_list),
        }
    export_path = run_dir / "export.json"
    export = json.loads(export_path.read_text(encoding="utf-8")) if export_path.is_file() else {}
    export["tflite"] = info
    export_path.write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--no-int8", action="store_true")
    args = parser.parse_args()
    info = export_tflite(args.run_dir, int8=not args.no_int8)
    print(json.dumps(info["files"], indent=2))


if __name__ == "__main__":
    main()

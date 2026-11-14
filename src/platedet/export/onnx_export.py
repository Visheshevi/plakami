"""Export a trained run to ONNX (FP32, opset 17, fixed input, raw head outputs, no NMS).

The output matches the official YOLOX ONNX files: (1, N, 5 + classes) with objectness and
class probabilities already passed through the sigmoid, boxes not yet decoded. Decoding and
NMS live in `platedet.postprocess`. Runs on CPU, so it works on the owner's Mac.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from platedet.eval.baseline import sha256_file
from platedet.train.exp import make_exp

OPSET = 17


def export_onnx(run_dir: Path, checkpoint: str = "yolox/best_ckpt.pth") -> Path:
    import torch

    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    size = int(config["input_size"])
    exp = make_exp(config, run_dir)
    model = exp.get_model()
    state = torch.load(run_dir / checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    model.head.decode_in_inference = False

    dummy = torch.zeros(1, 3, size, size)
    output = run_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["images"],
        output_names=["output"],
        opset_version=OPSET,
        dynamo=False,
    )

    # Check that ONNX Runtime reproduces PyTorch on a random input.
    import onnxruntime

    sample = torch.rand(1, 3, size, size) * 255
    with torch.no_grad():
        expected = model(sample).numpy()
    session = onnxruntime.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"images": sample.numpy()})[0]
    max_error = float(np.abs(actual - expected).max())
    if max_error > 1e-3:
        raise ValueError(f"ONNX output differs from PyTorch by {max_error}")

    info = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(run_dir / checkpoint),
        "onnx_sha256": sha256_file(output),
        "opset": OPSET,
        "input_size": size,
        "channel_order": config.get("channel_order", "bgr"),
        "max_abs_error_vs_torch": max_error,
    }
    (run_dir / "export.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="yolox/best_ckpt.pth")
    args = parser.parse_args()
    print(export_onnx(args.run_dir, args.checkpoint))


if __name__ == "__main__":
    main()

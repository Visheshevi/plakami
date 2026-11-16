# Third-party notices

This project and its model releases are licensed under the MIT License (`LICENSE`).
They include or derive from the third-party work below, which keeps its own licence.

## YOLOX (Apache-2.0)

- **What we use:** the YOLOX-Tiny model architecture, and the YOLOX-Tiny COCO checkpoint
  (`yolox_tiny.pth`, release 0.1.1rc0) as the starting point for training. The released
  model weights are a derivative of that checkpoint, fine-tuned for licence plates and
  vehicles. The training code uses the YOLOX source at commit
  `6ddff4824372906469a7fae2dc3206c7aa4bbaee`.
- **Source:** https://github.com/Megvii-BaseDetection/YOLOX
- **Copyright:** Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
- **Licence:** Apache License 2.0, full text in `licenses/YOLOX-Apache-2.0.txt`.
- **Changes:** the detection head was retrained for two classes (`license_plate`,
  `vehicle`), and the model was exported to ONNX and TFLite FP16.

## Training data

The weights were trained on images under CC BY 2.0 (Open Images V6, via Flickr),
MIT (CCPD) and CC0 1.0 (Kaggle "Car License Plate Detection"). Each image is credited in
`ATTRIBUTION.md` (in release packages; built from the dataset manifest). The test-vector
photos are credited in `test_vectors/ATTRIBUTION.md`.

## Runtime libraries (not included)

Running the models needs a runtime you install yourself, for example ONNX Runtime (MIT)
or LiteRT / TensorFlow Lite (Apache-2.0). The reference `postprocess.py` needs NumPy
(BSD-3-Clause) and Pillow (MIT-CMU).

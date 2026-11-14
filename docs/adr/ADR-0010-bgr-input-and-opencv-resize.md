# ADR-0010: BGR input and OpenCV-style resize for YOLOX-trained models

- **Status:** Accepted for implementation (owner chose BGR on 2026-11-12). The SPEC §5 wording change is pending owner approval.
- **Date:** 2026-11-12

## Context

- **Channel order.** SPEC §5 specifies RGB input. YOLOX's data loader reads images with OpenCV, so training images are BGR, and the official COCO weights were trained on BGR too. Feeding RGB would need a YOLOX patch, and it would mismatch the pretrained weights' first layer.
- **Resize.** YOLOX resizes with `cv2.INTER_LINEAR`, which has no antialiasing, and truncates the target size with `int()`. Our reference `letterbox` used Pillow's antialiased bilinear filter and rounded the size. That gives slightly different pixels at inference time than in training.

## Decision

- **BGR.** Models trained in this project take **BGR** input. `channel_order` is recorded in the training config, `export.json` and, later, `release.json`.
- **Resize.** `platedet.postprocess.letterbox` now truncates the target size like YOLOX and uses a NumPy bilinear resize with OpenCV `INTER_LINEAR` semantics: half-pixel centres, edge clamping and no antialiasing. A test checks that our output is within 1 grey level of YOLOX's own `cv2` pipeline. OpenCV's fixed-point rounding makes about 20% of pixels differ by exactly 1.
- **Documentation.** `docs/preprocessing.md` is updated to match.

## Consequences

- **Spec edit needed.** SPEC §5 says RGB. Proposed wording for the owner: *"Pre-processing: letterbox resize (top-left, pad 114, OpenCV-style bilinear), BGR channel order, float32 in 0–255, no mean/std."*
- **Baseline shift.** The T2 baseline numbers were measured with the old Pillow resize. `docs/baseline.md` records the rerun.
- **Ports.** Ports to other platforms must convert decoded images to BGR. The release test vectors (T5) will catch a port that forgets.

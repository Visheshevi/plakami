# ADR-0009: Pinned YOLOX source and a separate training dependency group

- **Status:** Accepted (owner approved the T3 plan on 2026-11-12)
- **Date:** 2026-11-12

## Context

SPEC §3 says to train with the official YOLOX code at a pinned commit. YOLOX can't be a normal locked dependency, because its `setup.py` imports `torch` at build time, which breaks isolated builds. YOLOX's trainer also requires CUDA: it hard-codes `cuda:` devices and `torch.cuda.set_device`.

## Decision

- **Pinned source.** `platedet.train.yolox_src` downloads YOLOX at commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee` (Apache-2.0, checked on fetch) into git-ignored `.cache/` and puts it on `sys.path`. `make setup` and `make test` fetch it. The commit SHA is the integrity anchor, and every run records it.
- **Training dependency group.** A uv `train` group holds the libraries training needs: `torch`, `torchvision`, `opencv-python-headless`, `loguru`, `pycocotools`, `tabulate`, `thop`, `psutil`, `tqdm`, `tensorboard` and `onnx`. Licences were checked on 2026-11-12; none is GPL or AGPL. `opencv-python-headless` bundles FFmpeg under LGPL, and `tqdm` is MPL-2.0 and MIT. `thop` is Lyken17's MIT package, not `ultralytics-thop`. The group is in `default-groups`, so `make setup` and `make test` install it.
- **Where training runs.** Full training runs on free Colab (owner-approved), using a bundle and notebook. Locally, `make test` runs a CPU smoke test: it builds the real YOLOX model on generated images, takes one training step, exports to ONNX and checks ONNX Runtime against PyTorch.
- **Export.** ONNX export uses the TorchScript exporter (`dynamo=False`, opset 17), which produces the same graph as the official YOLOX ONNX files. PyTorch marks this exporter as legacy, so T5 should recheck it.

## Consequences

`torch` makes the local environment larger. GPU training can't be tested in CI, so the smoke test covers the parts that matter for correctness: the dataset layout, the class mapping, loss computation and export. Upgrading YOLOX means changing one constant and rerunning `make test`.

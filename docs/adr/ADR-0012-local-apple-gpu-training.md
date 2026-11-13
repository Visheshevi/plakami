# ADR-0012: Train v0 locally on the Apple-silicon GPU

- **Status:** Accepted (owner chose local training on 2026-11-12, after a timing test)
- **Date:** 2026-11-12

## Context

T3 planned training on free Colab (ADR-0009), because YOLOX's `Trainer` only runs on CUDA. Colab needs a 2.6 GB upload, and its sessions disconnect. The owner's Mac (M2 Pro, 16 GB unified memory, 16-core GPU) has PyTorch MPS support.

A timing test used the real 640 data pipeline, mosaic augmentation and 200 steps per setup:

| Setup | Time per step | 100 epochs at 640 | Memory |
|---|---|---|---|
| Batch 16, fp32 | 0.66 s | ~11 h | 5.7 GB |
| Batch 16, fp16 autocast | 0.72 s | ~12 h | 5.7 GB |
| Batch 32, fp32 / fp16 | 6.5 s / 3.8 s | 54 h / 31 h | ~11 GB (memory pressure) |

The loss fell over the run (about 15 down to 9), and data loading takes 0.14 s per 32 images, so it isn't the bottleneck.

## Decision

- **`make train` picks the device.** It uses CUDA with YOLOX's own `Trainer` (unchanged, Colab) if available, otherwise the Apple GPU (MPS).
- **MPS uses `src/platedet/train/local_trainer.py`.** It subclasses YOLOX's `Trainer` and replaces only the CUDA-specific parts:
  - device setup;
  - the CUDA-stream prefetcher;
  - the multi-scale size tensor;
  - GPU-memory logging;
  - the COCO evaluator loop.

  The evaluator loop has the same predictions, postprocess and pycocotools scoring on val. The following stay YOLOX code: LR schedule, EMA, closing mosaic plus L1 loss for the last epochs, resume, checkpoint format and best-checkpoint selection on val AP50:95.
- **Type-name shim:** while training on MPS, `mps_type_names()` maps the `'torch.mps.FloatTensor'` type strings that YOLOX passes to `Tensor.type()`. It is restored afterwards.
- **MPS cache release:** the MPS allocator keeps cached buffers for every input size it has seen. With multi-scale training (480–800 px), the first run grew until macOS swapped about 14 GB and stalled at iteration 200. The trainer now calls `torch.mps.empty_cache()` at each size change and after each evaluation. Memory then follows the current size (about 4.4 GB at 480 px, 8.6 GB at 800 px), and speed stays at about 0.6–0.75 s per step.
- **`best_ckpt.pth` is also written at the first evaluation,** so a run always has one, even if val AP is still 0.
- **Model-summary log line:** it shows the parameter count only. YOLOX's FLOPs counter, `thop`, imports `distutils`, which Python 3.12 removed.
- **v0 configs:** `batch_size: 16` (LR scales per image, `basic_lr_per_img` is unchanged), `fp16: false`, `data_num_workers: 6`.
- **Run record:** `run_info.json` records the device, along with the device used for each resume.
- **`make train`** wraps the run in `caffeinate -i -s` on macOS. `RUN_ID=` sets a fixed run folder, so rerunning the same command resumes.

## Consequences

- No uploads and no session limits. Each run ties up the Mac for about 12 h (640) and about 5 h (416, estimated).
- Batch 16 rather than YOLOX's usual 64, or the 32 planned for Colab. It is recorded in the run config. BatchNorm statistics may be slightly noisier, and we'll revisit this in T4 if accuracy falls short.
- MPS and CUDA results are not bit-identical. Runs record the device, so any comparison should note it.
- The Colab notebook still works as a fallback, but `data/colab/platedet_bundle.tar` was built from the earlier commit. Rebuild it with `make colab-bundle` before using it, and set `data_num_workers: 2` on free Colab.
- Tests: CPU training, evaluation and resume, plus a 1-epoch Apple-GPU run that is skipped when no Apple GPU is present.

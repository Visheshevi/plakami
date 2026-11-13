"""YOLOX training on an Apple-silicon GPU (MPS) or the CPU (ADR-0012).

YOLOX's `Trainer` assumes CUDA in a few places: device selection, the CUDA-stream data
prefetcher, the multi-scale size tensor, GPU-memory logging and the COCO evaluator. This
subclass replaces only those parts. Everything else (LR schedule, EMA, closing mosaic for
the last epochs, L1 loss, resume, checkpoint names and contents) is YOLOX's own code, so a
local run and a Colab run produce the same kind of checkpoint.

YOLOX also passes tensor-type names around as strings (`x.type()` -> 'torch.mps.FloatTensor')
and `Tensor.type()` rejects the MPS names. `mps_type_names()` maps them while training runs.
"""

from __future__ import annotations

import contextlib
import datetime
import os
import random
import time
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import torch

_MPS_TYPE_NAMES = {"torch.mps.FloatTensor": torch.float32, "torch.mps.HalfTensor": torch.float16}


@contextlib.contextmanager
def mps_type_names() -> Iterator[None]:
    """Let `Tensor.type('torch.mps.FloatTensor')` work (YOLOX head, IoU and loss code)."""
    original = torch.Tensor.type

    def patched(self: torch.Tensor, dtype: Any = None, *args: Any, **kwargs: Any) -> Any:
        if isinstance(dtype, str) and dtype in _MPS_TYPE_NAMES:
            return self.to(device="mps", dtype=_MPS_TYPE_NAMES[dtype])
        return original(self, dtype, *args, **kwargs)

    torch.Tensor.type = patched  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.Tensor.type = original  # type: ignore[method-assign]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def memory_mb(device: str) -> float:
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return 0.0


class _Batches:
    """Stand-in for YOLOX's CUDA-stream `DataPrefetcher`: next batch, moved to the device."""

    def __init__(self, loader: Any, device: str) -> None:
        self.iterator = iter(loader)
        self.device = device

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        images, targets, _, _ = next(self.iterator)
        return images.to(self.device), targets.to(self.device)


def make_local_trainer(exp: Any, args: Any, device: str) -> Any:
    """Return a YOLOX `Trainer` subclass instance that runs on `device` ('mps' or 'cpu')."""
    from loguru import logger
    from torch.utils.tensorboard import SummaryWriter
    from yolox.core import Trainer
    from yolox.utils import (
        ModelEMA,
        adjust_status,
        mem_usage,
        postprocess,
    )

    class LocalTrainer(Trainer):  # type: ignore[misc]
        def __init__(self) -> None:
            if args.fp16:
                raise ValueError("fp16 is CUDA-only here; set train.fp16: false")
            super().__init__(exp, args)
            self.device = device

        def before_train(self) -> None:
            logger.info(f"args: {self.args}")
            logger.info(f"exp value:\n{self.exp}")
            model = self.exp.get_model()
            # YOLOX's get_model_info needs thop, which needs distutils (gone in Python 3.12).
            params = sum(parameter.numel() for parameter in model.parameters()) / 1e6
            logger.info(f"Model Summary: Params: {params:.2f}M")
            model.to(self.device)
            self.optimizer = self.exp.get_optimizer(self.args.batch_size)
            model = self.resume_train(model)
            self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
            self.train_loader = self.exp.get_data_loader(
                batch_size=self.args.batch_size,
                is_distributed=False,
                no_aug=self.no_aug,
                cache_img=None,
            )
            self.prefetcher = _Batches(self.train_loader, self.device)
            self.max_iter = len(self.train_loader)
            self.lr_scheduler = self.exp.get_lr_scheduler(
                self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
            )
            if self.use_model_ema:
                self.ema_model = ModelEMA(model, 0.9998)
                self.ema_model.updates = self.max_iter * self.start_epoch
            self.model = model
            self.evaluator = self.exp.get_evaluator(
                batch_size=self.args.batch_size, is_distributed=False
            )
            self.tblogger = SummaryWriter(f"{self.file_name}/tensorboard")
            logger.info(f"Training start on {self.device}...")

        def train_one_iter(self) -> None:
            start = time.time()
            inps, targets = self.prefetcher.next()
            inps = inps.to(self.data_type)
            targets = targets.to(self.data_type)
            targets.requires_grad = False
            inps, targets = self.exp.preprocess(inps, targets, self.input_size)
            data_end = time.time()
            outputs = self.model(inps, targets)
            loss = outputs["total_loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss is {loss.item()} at iter {self.progress_in_iter}")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.use_model_ema:
                self.ema_model.update(self.model)
            lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.meter.update(
                iter_time=time.time() - start, data_time=data_end - start, lr=lr, **outputs
            )

        def after_iter(self) -> None:
            if (self.iter + 1) % self.exp.print_interval == 0:
                left = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
                eta = datetime.timedelta(seconds=int(self.meter["iter_time"].global_avg * left))
                losses = self.meter.get_filtered_meter("loss")
                times = self.meter.get_filtered_meter("time")
                logger.info(
                    f"epoch: {self.epoch + 1}/{self.max_epoch}, "
                    f"iter: {self.iter + 1}/{self.max_iter}, "
                    f"{self.device} mem: {memory_mb(self.device):.0f}Mb, "
                    f"mem: {mem_usage():.1f}Gb, "
                    + ", ".join(f"{k}: {v.avg:.3f}s" for k, v in times.items())
                    + ", "
                    + ", ".join(f"{k}: {v.latest:.1f}" for k, v in losses.items())
                    + f", lr: {self.meter['lr'].latest:.3e}, size: {self.input_size[0]}, ETA: {eta}"
                )
                self.tblogger.add_scalar("train/lr", self.meter["lr"].latest, self.progress_in_iter)
                for key, value in losses.items():
                    self.tblogger.add_scalar(f"train/{key}", value.latest, self.progress_in_iter)
                self.meter.clear_meters()
            if (self.progress_in_iter + 1) % 10 == 0:
                self.input_size = self._random_size()

        def _random_size(self) -> tuple[int, int]:
            """`Exp.random_resize` without its CUDA tensor (single process)."""
            if not hasattr(self.exp, "random_size"):
                centre = int(self.exp.input_size[0] / 32)
                self.exp.random_size = (
                    centre - self.exp.multiscale_range,
                    centre + self.exp.multiscale_range,
                )
            size = random.randint(*self.exp.random_size)
            factor = self.exp.input_size[1] / self.exp.input_size[0]
            return int(32 * size), 32 * int(size * factor)

        def evaluate_and_save_model(self) -> None:
            model = self.ema_model.ema if self.use_model_ema else self.model
            with adjust_status(model, training=False):
                (ap50_95, ap50, summary), _ = self._evaluate(model)
            # Also write best_ckpt.pth on the first evaluation, even at AP 0.
            first = not os.path.exists(os.path.join(self.file_name, "best_ckpt.pth"))
            update_best = ap50_95 > self.best_ap or first
            self.best_ap = max(self.best_ap, ap50_95)
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
            self.save_ckpt("last_epoch", update_best, ap=ap50_95)

        def _evaluate(self, model: Any) -> Any:
            """`COCOEvaluator.evaluate` on this device (val split only; selects best_ckpt)."""
            evaluator = self.evaluator
            data_list: list[Any] = []
            outputs_by_image: dict[Any, Any] = defaultdict(dict)
            inference_time = nms_time = 0.0
            for images, _, info_imgs, ids in evaluator.dataloader:
                with torch.no_grad():
                    start = time.time()
                    outputs = model(images.to(self.device).float())
                    outputs = outputs.cpu()
                    inferred = time.time()
                    outputs = postprocess(
                        outputs, evaluator.num_classes, evaluator.confthre, evaluator.nmsthre
                    )
                    inference_time += inferred - start
                    nms_time += time.time() - inferred
                elements, per_image = evaluator.convert_to_coco_format(
                    outputs, info_imgs, ids, return_outputs=True
                )
                data_list.extend(elements)
                outputs_by_image.update(per_image)
            statistics = torch.tensor(
                [inference_time, nms_time, max(len(evaluator.dataloader), 1)], dtype=torch.float32
            )
            return evaluator.evaluate_prediction(data_list, statistics), outputs_by_image

    return LocalTrainer()

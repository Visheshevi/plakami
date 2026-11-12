"""YOLOX experiment built from a training YAML (two classes, our COCO layout).

Defaults follow the official `exps/default/yolox_tiny.py`; the YAML can override the
training schedule. Images are read as `<training_dir>/<file_name>` (no COCO split folder).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from platedet.train.yolox_src import ensure_yolox

TINY_DEPTH = 0.33
TINY_WIDTH = 0.375
NUM_CLASSES = 2


def make_exp(config: dict[str, Any], output_dir: Path) -> Any:
    """Return a configured `yolox.exp.Exp` instance."""
    ensure_yolox()
    from yolox.data import COCODataset, TrainTransform, ValTransform
    from yolox.exp import Exp

    training = config.get("train", {})
    size = int(config["input_size"])
    data_dir = str(Path(config["data"]["training_dir"]).resolve())

    class PlateExp(Exp):
        def __init__(self) -> None:
            super().__init__()
            self.depth = TINY_DEPTH
            self.width = TINY_WIDTH
            self.num_classes = NUM_CLASSES
            self.input_size = (size, size)
            self.test_size = (size, size)
            self.mosaic_scale = (0.5, 1.5)
            self.enable_mixup = False
            self.data_dir = data_dir
            self.train_ann = "train.json"
            self.val_ann = "val.json"
            self.output_dir = str(output_dir)
            self.exp_name = str(config["name"])
            self.seed = int(config.get("seed", 0))
            self.save_history_ckpt = False
            self.nmsthre = float(config.get("nms_iou", 0.45))
            if "random_size" in training:
                self.random_size = tuple(training["random_size"])
            for key in (
                "max_epoch",
                "no_aug_epochs",
                "warmup_epochs",
                "eval_interval",
                "data_num_workers",
                "basic_lr_per_img",
                "print_interval",
            ):
                if key in training:
                    setattr(self, key, type(getattr(self, key))(training[key]))
            if "mosaic_scale" in training:
                self.mosaic_scale = tuple(training["mosaic_scale"])

        def get_dataset(self, cache: bool = False, cache_type: str = "ram") -> Any:
            return COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                name="",
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
                ),
                cache=cache,
                cache_type=cache_type,
            )

        def get_eval_dataset(self, **kwargs: Any) -> Any:
            return COCODataset(
                data_dir=self.data_dir,
                json_file=self.val_ann,
                name="",
                img_size=self.test_size,
                preproc=ValTransform(legacy=False),
            )

    return PlateExp()

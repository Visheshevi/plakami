import math

import numpy as np
import pytest
from PIL import Image

from platedet.postprocess import (
    PAD_VALUE,
    decode_outputs,
    letterbox,
    nms,
    postprocess,
)

SIZE = 64  # grids 8x8 + 4x4 + 2x2 = 84 predictions


def _raw(num_classes: int = 2) -> np.ndarray:
    return np.zeros((84, 5 + num_classes), dtype=np.float32)


def _set(raw, index, *, stride, cell_x, cell_y, w, h, obj, cls):
    """Write a raw prediction that decodes to a box centred in (cell_x, cell_y)."""
    raw[index, 0:2] = 0.5, 0.5
    raw[index, 2:4] = math.log(w / stride), math.log(h / stride)
    raw[index, 4] = obj
    raw[index, 5:] = cls


def test_letterbox_keeps_ratio_and_pads_bottom_right():
    tensor, ratio = letterbox(Image.new("RGB", (200, 100), (255, 0, 0)), SIZE)
    assert tensor.shape == (1, 3, SIZE, SIZE) and tensor.dtype == np.float32
    assert ratio == pytest.approx(0.32)
    assert (tensor[0, 0, :32, :] == 255).all()
    assert (tensor[0, :, 32:, :] == PAD_VALUE).all()


def test_letterbox_bgr_reverses_channels():
    rgb, _ = letterbox(Image.new("RGB", (8, 8), (255, 0, 0)), 8)
    bgr, _ = letterbox(Image.new("RGB", (8, 8), (255, 0, 0)), 8, "bgr")
    assert rgb[0, 0, 0, 0] == 255 and bgr[0, 2, 0, 0] == 255 and bgr[0, 0, 0, 0] == 0
    with pytest.raises(ValueError):
        letterbox(Image.new("RGB", (8, 8)), 8, "hsv")


def test_decode_uses_grid_offsets_and_strides():
    raw = _raw()
    # Stride-8 grid is 8x8; cell (x=1, y=2) is row 2 * 8 + 1.
    _set(raw, 17, stride=8, cell_x=1, cell_y=2, w=16, h=8, obj=1, cls=[1, 0])
    # First stride-16 cell (index 64) and first stride-32 cell (index 80).
    _set(raw, 80, stride=32, cell_x=0, cell_y=0, w=32, h=32, obj=1, cls=[0, 1])
    decoded = decode_outputs(raw, SIZE)
    np.testing.assert_allclose(decoded[17, :4], [12, 20, 16, 8], rtol=1e-5)
    np.testing.assert_allclose(decoded[80, :4], [16, 16, 32, 32], rtol=1e-5)
    with pytest.raises(ValueError):
        decode_outputs(np.zeros((10, 7), dtype=np.float32), SIZE)


def test_nms_keeps_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    assert nms(boxes, np.array([0.5, 0.9, 0.7]), 0.45) == [1, 2]


def test_postprocess_scales_back_scores_and_is_class_aware():
    raw = _raw()
    _set(raw, 17, stride=8, cell_x=1, cell_y=2, w=16, h=8, obj=0.9, cls=[0.8, 0.1])
    _set(raw, 18, stride=8, cell_x=2, cell_y=2, w=16, h=8, obj=0.5, cls=[0.9, 0.1])
    raw[18, 0] = -0.3  # centre x 13.6: IoU 0.82 with index 17, so NMS drops it
    _set(raw, 26, stride=8, cell_x=2, cell_y=3, w=16, h=8, obj=0.9, cls=[0.1, 0.9])  # other class
    detections = postprocess(
        raw[None], size=SIZE, ratio=0.5, image_width=100, image_height=100, conf_threshold=0.3
    )
    assert [(d.class_id, round(d.score, 2)) for d in detections] == [(2, 0.81), (1, 0.72)]
    plate = detections[1]
    assert (plate.x1, plate.y1, plate.x2, plate.y2) == pytest.approx((8, 32, 40, 48))


def test_postprocess_class_map_drops_unmapped_and_merges_before_nms():
    raw = _raw(num_classes=3)
    _set(raw, 17, stride=8, cell_x=1, cell_y=2, w=16, h=8, obj=1, cls=[0.9, 0, 0])  # "car"
    _set(raw, 18, stride=8, cell_x=1, cell_y=2, w=16, h=8, obj=1, cls=[0, 0.8, 0])  # "truck"
    _set(raw, 40, stride=8, cell_x=0, cell_y=5, w=16, h=8, obj=1, cls=[0, 0, 0.9])  # unmapped
    raw[18, 0] = -0.4  # centre x 12.8: IoU 0.9 with index 17
    detections = postprocess(
        raw,
        size=SIZE,
        ratio=1.0,
        image_width=SIZE,
        image_height=SIZE,
        conf_threshold=0.1,
        class_map={0: 2, 1: 2},
    )
    assert len(detections) == 1 and detections[0].class_id == 2
    assert detections[0].score == pytest.approx(0.9)


def test_postprocess_clips_to_image_and_handles_empty():
    raw = _raw()
    _set(raw, 80, stride=32, cell_x=0, cell_y=0, w=64, h=64, obj=1, cls=[1, 0])
    (detection,) = postprocess(
        raw, size=SIZE, ratio=1.0, image_width=40, image_height=30, conf_threshold=0.5
    )
    assert (detection.x1, detection.y1, detection.x2, detection.y2) == (0, 0, 40, 30)
    assert (
        postprocess(
            _raw(), size=SIZE, ratio=1.0, image_width=40, image_height=30, conf_threshold=0.5
        )
        == []
    )

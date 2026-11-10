"""Stable image-level dataset splitting."""

from __future__ import annotations

import hashlib

DEFAULT_SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def validate_split_ratios(ratios: dict[str, float]) -> None:
    if set(ratios) != {"train", "val", "test"}:
        raise ValueError("Split ratios must define exactly train, val, and test")
    if any(value < 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1) > 1e-9:
        raise ValueError("Split ratios must be non-negative and sum to 1")


def assign_split(image_id: str, ratios: dict[str, float] | None = None) -> str:
    """Assign an image to a stable split by SHA-256, independent of input order."""
    selected_ratios = ratios or DEFAULT_SPLIT_RATIOS
    validate_split_ratios(selected_ratios)
    value = int.from_bytes(hashlib.sha256(image_id.encode("utf-8")).digest()[:8], "big") / 2**64
    train_end = selected_ratios["train"]
    val_end = train_end + selected_ratios["val"]
    if value < train_end:
        return "train"
    if value < val_end:
        return "val"
    return "test"

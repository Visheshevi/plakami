"""Perceptual-hash deduplication using a BK-tree for Hamming distance lookup."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


@dataclass
class _Node:
    value: str
    children: dict[int, _Node] = field(default_factory=dict)


class _BKTree:
    def __init__(self) -> None:
        self.root: _Node | None = None

    def add(self, value: str) -> None:
        if self.root is None:
            self.root = _Node(value)
            return
        node = self.root
        while True:
            distance = hamming_distance(value, node.value)
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _Node(value)
                return
            node = child

    def contains_near(self, value: str, threshold: int) -> bool:
        if self.root is None:
            return False
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = hamming_distance(value, node.value)
            if distance <= threshold:
                return True
            low, high = distance - threshold, distance + threshold
            pending.extend(child for edge, child in node.children.items() if low <= edge <= high)
        return False


def deduplicate_by_phash[T](
    records: list[T], *, phash_getter: Callable[[T], str], threshold: int
) -> tuple[list[T], int]:
    """Keep the first record per perceptual-hash neighborhood; order is caller-defined."""
    if threshold < 0:
        raise ValueError("Perceptual hash threshold cannot be negative")
    tree = _BKTree()
    kept: list[T] = []
    duplicates = 0
    for record in records:
        phash = phash_getter(record)
        if tree.contains_near(phash, threshold):
            duplicates += 1
            continue
        tree.add(phash)
        kept.append(record)
    return kept, duplicates

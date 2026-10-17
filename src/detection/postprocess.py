"""Class-aware confidence filtering and class-agnostic duplicate suppression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    xyxy: tuple[int, int, int, int]
    track_id: int | None = None


def _intersection_and_areas(
    first: Detection, second: Detection
) -> tuple[float, float, float]:
    ax1, ay1, ax2, ay2 = first.xyxy
    bx1, by1, bx2, by2 = second.xyxy
    intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = float(intersection_width * intersection_height)
    first_area = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    second_area = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    return intersection, first_area, second_area


def overlap_scores(first: Detection, second: Detection) -> tuple[float, float]:
    """Return IoU and intersection-over-smaller-box (IoS)."""
    intersection, first_area, second_area = _intersection_and_areas(first, second)
    union = first_area + second_area - intersection
    iou = intersection / union if union > 0 else 0.0
    smaller_area = min(first_area, second_area)
    ios = intersection / smaller_area if smaller_area > 0 else 0.0
    return iou, ios


def filter_detections(
    detections: Iterable[Detection],
    *,
    default_confidence: float,
    class_confidence: Mapping[int, float],
    duplicate_iou: float,
    duplicate_ios: float,
) -> list[Detection]:
    """Apply per-class thresholds, then remove same-object boxes across classes."""
    candidates = [
        detection
        for detection in detections
        if detection.confidence
        >= class_confidence.get(detection.class_id, default_confidence)
    ]
    kept: list[Detection] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        for accepted in kept:
            iou, ios = overlap_scores(candidate, accepted)
            if iou >= duplicate_iou or ios >= duplicate_ios:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept

"""Pixel-preserving numbered overlays shared by both graph implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import cv2
import numpy as np


class PixelBox(Protocol):
    """Pixel bounds read by the overlay rasterizer."""

    @property
    def x0(self) -> int:
        """Left pixel bound."""
        ...

    @property
    def y0(self) -> int:
        """Top pixel bound."""
        ...

    @property
    def x1(self) -> int:
        """Right pixel bound."""
        ...

    @property
    def y1(self) -> int:
        """Bottom pixel bound."""
        ...


def draw_numbered_boxes(
    image_bgr: np.ndarray,
    boxes: Sequence[PixelBox],
    labels: list[str],
    *,
    palette: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    """The declutter counterpart of ``draw_boxes``: distinct colors + collision-avoided pins.

    Thin same-color track-id text stacks unreadably where boxes overlap (measured in
    reasoning_agent; its declutter plan is the design source). Each box gets a DISTINCT
    palette color shared by outline and pin -- the color binds the pair -- with the pin
    anchored inside the top-left corner, nudged downward past already-placed pins, and
    a leader line back to its corner when nudged.
    """
    annotated = image_bgr.copy()
    height = annotated.shape[0]
    placed: list[tuple[int, int, int]] = []
    for index, (box, label) in enumerate(zip(boxes, labels, strict=True)):
        slot = int(label) - 1 if label.isdigit() else index
        color = palette[slot % len(palette)]
        cv2.rectangle(annotated, (box.x0, box.y0), (box.x1, box.y1), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        radius = max(text_w, text_h) // 2 + 4
        anchor_xy = (box.x0 + radius, min(max(box.y0 + radius, radius), height - radius))
        cx, cy = anchor_xy
        while any(
            (cx - px) ** 2 + (cy - py) ** 2 < (radius + pr) ** 2 for px, py, pr in placed
        ) and cy + 2 * radius < min(box.y1, height - radius):
            cy += 2 * radius
        if (cx, cy) != anchor_xy:
            cv2.line(annotated, (cx, cy), (box.x0 + 2, box.y0 + 2), color, 2, cv2.LINE_AA)
        placed.append((cx, cy, radius))
        draw_pin(annotated, (cx, cy), label, color)
    return annotated


def draw_pin(
    image_bgr: np.ndarray, center: tuple[int, int], label: str, color: tuple[int, int, int]
) -> None:
    """Draw a filled ``color`` disc at ``center`` carrying ``label`` legibly, in place.

    The opaque disc gives the text a consistent background regardless of underlying pixels; the
    label is white over a black outline (the contrast pattern used for grounded reference points)
    so it reads over the disc and any overlapping fills.
    """
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.circle(image_bgr, center, max(text_w, text_h) // 2 + 4, color, -1)
    origin = (center[0] - text_w // 2, center[1] + text_h // 2)
    cv2.putText(image_bgr, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(
        image_bgr, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
    )

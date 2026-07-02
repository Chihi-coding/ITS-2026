"""Geometry and visualization helpers for the AI module."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_roi_config(config_path: str | Path) -> dict[str, Any]:
    """Load ROI polygon configuration from JSON."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"ROI config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    polygon = data.get("polygon", data)
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError("ROI polygon must contain at least 3 coordinate pairs")

    data["polygon"] = np.array(polygon, dtype=np.int32)
    data["max_dwell_seconds"] = float(data.get("max_dwell_seconds", 10))
    return data


def bbox_bottom_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Return the bottom-center point of a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, float(y2)


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """Return True when point lies inside or on the ROI polygon boundary."""
    result = cv2.pointPolygonTest(polygon, point, False)
    return result >= 0


def draw_roi(frame: np.ndarray, polygon: np.ndarray, color: tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """Draw the ROI polygon overlay on a frame."""
    overlay = frame.copy()
    cv2.polylines(overlay, [polygon], isClosed=True, color=color, thickness=2)
    cv2.fillPoly(overlay, [polygon], color=(0, 0, 255))
    return cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)


def draw_detection(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw a bounding box and label on the frame in-place."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 8, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )

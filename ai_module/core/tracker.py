"""Vehicle tracking with ROI dwell-time violation logic."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ai_module.utils.helpers import bbox_bottom_center, point_in_polygon

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    enter_time: float | None = None
    last_seen: float = field(default_factory=time.time)
    is_in_roi: bool = False
    is_reported: bool = False
    dwell_seconds: float = 0.0


@dataclass
class ViolationEvent:
    track_id: int
    bbox: tuple[int, int, int, int]
    dwell_seconds: float
    timestamp: float


class VehicleTracker:
    """
    Lightweight SORT-style tracker with ROI dwell monitoring.

    Uses IoU matching between frames and cv2.pointPolygonTest for ROI checks.
    """

    def __init__(
        self,
        roi_polygon: np.ndarray,
        max_dwell_seconds: float = 10.0,
        iou_threshold: float = 0.3,
        max_missing_seconds: float = 2.0,
    ) -> None:
        self.roi_polygon = roi_polygon
        self.max_dwell_seconds = max_dwell_seconds
        self.iou_threshold = iou_threshold
        self.max_missing_seconds = max_missing_seconds
        self.tracks: dict[int, TrackState] = {}
        self._next_id = 1

    @staticmethod
    def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _assign_tracks(self, detections: list[tuple[int, int, int, int]]) -> dict[int, tuple[int, int, int, int]]:
        """Greedy IoU matching between existing tracks and current detections."""
        assignments: dict[int, tuple[int, int, int, int]] = {}
        unmatched_detections = list(detections)
        now = time.time()

        active_track_ids = [
            track_id
            for track_id, state in self.tracks.items()
            if now - state.last_seen <= self.max_missing_seconds
        ]

        for track_id in active_track_ids:
            best_iou = 0.0
            best_det: tuple[int, int, int, int] | None = None
            state = self.tracks[track_id]

            for det in unmatched_detections:
                score = self._iou(state.bbox, det)
                if score > best_iou:
                    best_iou = score
                    best_det = det

            if best_det is not None and best_iou >= self.iou_threshold:
                assignments[track_id] = best_det
                unmatched_detections.remove(best_det)

        for det in unmatched_detections:
            track_id = self._next_id
            self._next_id += 1
            assignments[track_id] = det
            self.tracks[track_id] = TrackState(track_id=track_id, bbox=det)

        return assignments

    def update(self, detections: list[tuple[int, int, int, int]]) -> tuple[dict[int, TrackState], list[ViolationEvent]]:
        """
        Update track states and emit violation events when dwell time exceeds threshold.

        Returns updated track dictionary and a list of new violation events.
        """
        now = time.time()
        assignments = self._assign_tracks(detections)
        violations: list[ViolationEvent] = []

        seen_ids = set(assignments.keys())
        for track_id, bbox in assignments.items():
            state = self.tracks.setdefault(track_id, TrackState(track_id=track_id, bbox=bbox))
            state.bbox = bbox
            state.last_seen = now

            center = bbox_bottom_center(bbox)
            inside = point_in_polygon(center, self.roi_polygon)

            if inside:
                if state.enter_time is None:
                    state.enter_time = now
                    logger.debug("Track %s entered ROI", track_id)
                state.is_in_roi = True
                state.dwell_seconds = now - state.enter_time

                if (
                    state.dwell_seconds >= self.max_dwell_seconds
                    and not state.is_reported
                ):
                    state.is_reported = True
                    violations.append(
                        ViolationEvent(
                            track_id=track_id,
                            bbox=bbox,
                            dwell_seconds=state.dwell_seconds,
                            timestamp=now,
                        )
                    )
                    logger.info(
                        "Violation detected for track %s after %.1fs in ROI",
                        track_id,
                        state.dwell_seconds,
                    )
            else:
                if state.is_in_roi:
                    logger.debug("Track %s left ROI; resetting dwell state", track_id)
                state.is_in_roi = False
                state.enter_time = None
                state.dwell_seconds = 0.0
                state.is_reported = False

        stale_ids = [
            track_id
            for track_id, state in self.tracks.items()
            if track_id not in seen_ids and now - state.last_seen > self.max_missing_seconds
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]

        return self.tracks, violations

    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """Visualize active tracks and ROI membership."""
        output = frame.copy()
        for state in self.tracks.values():
            x1, y1, x2, y2 = state.bbox
            color = (0, 0, 255) if state.is_in_roi else (0, 255, 0)
            label = f"ID {state.track_id} | {state.dwell_seconds:.1f}s"
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                output,
                label,
                (x1, max(y1 - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
        return output
